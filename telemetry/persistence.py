#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRM Telemetry Persistence Module

This module provides data persistence functionality for the HRM system.
It handles saving and loading of system state, configuration, and historical data.
"""

import json
import pickle
import asyncio
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from dataclasses import asdict

from core.logging import logger


class DataPersistence:
    """Data persistence manager for HRM system."""
    
    def __init__(self, data_dir: str = "storage"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Subdirectories for different data types
        self.state_dir = self.data_dir / "state"
        self.config_dir = self.data_dir / "config"
        self.history_dir = self.data_dir / "history"
        self.backup_dir = self.data_dir / "backup"
        
        for directory in [self.state_dir, self.config_dir, self.history_dir, self.backup_dir]:
            directory.mkdir(exist_ok=True)
        
        # Persistence settings
        self.auto_save_interval = 300  # 5 minutes
        self.backup_interval = 3600    # 1 hour
        self.max_history_entries = 10000
        
        # Background tasks
        self._save_task = None
        self._backup_task = None
    
    async def start_persistence(self):
        """Start background persistence tasks."""
        if self._save_task is None:
            self._save_task = asyncio.create_task(self._auto_save_loop())
            logger.info("💾 Auto-save started")
        
        if self._backup_task is None:
            self._backup_task = asyncio.create_task(self._backup_loop())
            logger.info("💾 Backup system started")
    
    async def stop_persistence(self):
        """Stop background persistence tasks."""
        if self._save_task:
            self._save_task.cancel()
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass
            self._save_task = None
        
        if self._backup_task:
            self._backup_task.cancel()
            try:
                await self._backup_task
            except asyncio.CancelledError:
                pass
            self._backup_task = None
    
    async def _auto_save_loop(self):
        """Background auto-save loop."""
        while True:
            try:
                await asyncio.sleep(self.auto_save_interval)
                await self._perform_auto_save()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Auto-save error: {e}")
    
    async def _backup_loop(self):
        """Background backup loop."""
        while True:
            try:
                await asyncio.sleep(self.backup_interval)
                await self._perform_backup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Backup error: {e}")
    
    async def _perform_auto_save(self):
        """Perform auto-save of current state."""
        try:
            # Save system state
            await self.save_state("auto_save", self._get_current_state())
            
            # Save configuration
            await self.save_config("auto_save", self._get_current_config())
            
            logger.debug("💾 Auto-save completed")
            
        except Exception as e:
            logger.error(f"❌ Auto-save failed: {e}")
    
    async def _perform_backup(self):
        """Perform backup of all data."""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            # Create backup directory
            backup_path = self.backup_dir / f"backup_{timestamp}"
            backup_path.mkdir(exist_ok=True)
            
            # Backup state files
            for state_file in self.state_dir.glob("*.json"):
                backup_file = backup_path / f"state_{state_file.name}"
                await self._copy_file(state_file, backup_file)
            
            # Backup config files
            for config_file in self.config_dir.glob("*.json"):
                backup_file = backup_path / f"config_{config_file.name}"
                await self._copy_file(config_file, backup_file)
            
            # Backup history files
            for history_file in self.history_dir.glob("*.json"):
                backup_file = backup_path / f"history_{history_file.name}"
                await self._copy_file(history_file, backup_file)
            
            logger.info(f"💾 Backup completed: {backup_path}")
            
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
    
    async def _copy_file(self, source: Path, destination: Path):
        """Copy a file asynchronously."""
        try:
            with open(source, 'r', encoding='utf-8') as src:
                content = src.read()
            with open(destination, 'w', encoding='utf-8') as dst:
                dst.write(content)
        except Exception as e:
            logger.error(f"❌ File copy failed: {source} -> {destination}: {e}")
    
    async def save_state(self, name: str, state: Dict[str, Any]):
        """Save system state to persistent storage."""
        try:
            state_file = self.state_dir / f"{name}.json"
            
            # Add metadata
            state_with_metadata = {
                'timestamp': datetime.utcnow().isoformat(),
                'version': '1.0',
                'data': state
            }
            
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state_with_metadata, f, indent=2, ensure_ascii=False, default=str)
            
            logger.debug(f"💾 State saved: {name}")
            
        except Exception as e:
            logger.error(f"❌ State save failed: {e}")
    
    async def load_state(self, name: str) -> Optional[Dict[str, Any]]:
        """Load system state from persistent storage."""
        try:
            state_file = self.state_dir / f"{name}.json"
            
            if not state_file.exists():
                logger.warning(f"⚠️ State file not found: {name}")
                return None
            
            with open(state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.debug(f"📂 State loaded: {name}")
            return data.get('data')
            
        except Exception as e:
            logger.error(f"❌ State load failed: {e}")
            return None
    
    async def save_config(self, name: str, config: Dict[str, Any]):
        """Save configuration to persistent storage."""
        try:
            config_file = self.config_dir / f"{name}.json"
            
            # Add metadata
            config_with_metadata = {
                'timestamp': datetime.utcnow().isoformat(),
                'version': '1.0',
                'data': config
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_with_metadata, f, indent=2, ensure_ascii=False, default=str)
            
            logger.debug(f"💾 Config saved: {name}")
            
        except Exception as e:
            logger.error(f"❌ Config save failed: {e}")
    
    async def load_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Load configuration from persistent storage."""
        try:
            config_file = self.config_dir / f"{name}.json"
            
            if not config_file.exists():
                logger.warning(f"⚠️ Config file not found: {name}")
                return None
            
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.debug(f"📂 Config loaded: {name}")
            return data.get('data')
            
        except Exception as e:
            logger.error(f"❌ Config load failed: {e}")
            return None
    
    async def save_history(self, name: str, data: Union[List[Dict], pd.DataFrame]):
        """Save historical data to persistent storage."""
        try:
            history_file = self.history_dir / f"{name}.json"
            
            # Convert DataFrame to list of dicts if needed
            if isinstance(data, pd.DataFrame):
                data = data.to_dict('records')
            
            # Load existing history
            existing_data = []
            if history_file.exists():
                with open(history_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            
            # Append new data
            existing_data.extend(data)
            
            # Limit history size
            if len(existing_data) > self.max_history_entries:
                existing_data = existing_data[-self.max_history_entries:]
            
            # Save updated history
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.debug(f"💾 History saved: {name} ({len(data)} entries)")
            
        except Exception as e:
            logger.error(f"❌ History save failed: {e}")
    
    async def load_history(self, name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load historical data from persistent storage."""
        try:
            history_file = self.history_dir / f"{name}.json"
            
            if not history_file.exists():
                logger.warning(f"⚠️ History file not found: {name}")
                return []
            
            with open(history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Apply limit if specified
            if limit and len(data) > limit:
                data = data[-limit:]
            
            logger.debug(f"📂 History loaded: {name} ({len(data)} entries)")
            return data
            
        except Exception as e:
            logger.error(f"❌ History load failed: {e}")
            return []
    
    async def save_pickle(self, name: str, data: Any):
        """Save data using pickle serialization."""
        try:
            pickle_file = self.data_dir / f"{name}.pkl"
            
            with open(pickle_file, 'wb') as f:
                pickle.dump(data, f)
            
            logger.debug(f"💾 Pickle saved: {name}")
            
        except Exception as e:
            logger.error(f"❌ Pickle save failed: {e}")
    
    async def load_pickle(self, name: str) -> Optional[Any]:
        """Load data using pickle deserialization."""
        try:
            pickle_file = self.data_dir / f"{name}.pkl"
            
            if not pickle_file.exists():
                logger.warning(f"⚠️ Pickle file not found: {name}")
                return None
            
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
            
            logger.debug(f"📂 Pickle loaded: {name}")
            return data
            
        except Exception as e:
            logger.error(f"❌ Pickle load failed: {e}")
            return None
    
    def _get_current_state(self) -> Dict[str, Any]:
        """Get current system state for saving."""
        # This would typically collect state from various system components
        return {
            'system_info': {
                'timestamp': datetime.utcnow().isoformat(),
                'version': '1.0'
            }
        }
    
    def _get_current_config(self) -> Dict[str, Any]:
        """Get current system configuration for saving."""
        # This would typically collect config from various system components
        return {
            'system_config': {
                'timestamp': datetime.utcnow().isoformat(),
                'version': '1.0'
            }
        }
    
    async def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data files."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            # Clean up state files
            for state_file in self.state_dir.glob("*.json"):
                if state_file.stat().st_mtime < cutoff_date.timestamp():
                    state_file.unlink()
                    logger.info(f"🧹 Cleaned up old state file: {state_file}")
            
            # Clean up config files
            for config_file in self.config_dir.glob("*.json"):
                if config_file.stat().st_mtime < cutoff_date.timestamp():
                    config_file.unlink()
                    logger.info(f"🧹 Cleaned up old config file: {config_file}")
            
            # Clean up history files
            for history_file in self.history_dir.glob("*.json"):
                if history_file.stat().st_mtime < cutoff_date.timestamp():
                    history_file.unlink()
                    logger.info(f"🧹 Cleaned up old history file: {history_file}")
            
            # Clean up backup directories
            for backup_dir in self.backup_dir.glob("backup_*"):
                if backup_dir.stat().st_mtime < cutoff_date.timestamp():
                    import shutil
                    shutil.rmtree(backup_dir)
                    logger.info(f"🧹 Cleaned up old backup directory: {backup_dir}")
                    
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage usage statistics."""
        try:
            stats = {
                'timestamp': datetime.utcnow().isoformat(),
                'directories': {},
                'total_files': 0,
                'total_size_mb': 0.0
            }
            
            # Calculate stats for each directory
            for directory in [self.state_dir, self.config_dir, self.history_dir, self.backup_dir]:
                files = list(directory.glob("*"))
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                
                stats['directories'][directory.name] = {
                    'file_count': len(files),
                    'size_mb': total_size / (1024 * 1024),
                    'latest_file': max(files, key=lambda f: f.stat().st_mtime).name if files else None
                }
                
                stats['total_files'] += len(files)
                stats['total_size_mb'] += total_size / (1024 * 1024)
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Storage stats failed: {e}")
            return {}
    
    async def verify_integrity(self) -> Dict[str, Any]:
        """Verify data integrity across all storage."""
        try:
            integrity_report = {
                'timestamp': datetime.utcnow().isoformat(),
                'state_files': {},
                'config_files': {},
                'history_files': {},
                'overall_status': 'unknown'
            }
            
            # Verify state files
            for state_file in self.state_dir.glob("*.json"):
                try:
                    with open(state_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    integrity_report['state_files'][state_file.name] = {
                        'status': 'valid',
                        'timestamp': data.get('timestamp', 'unknown'),
                        'version': data.get('version', 'unknown')
                    }
                except Exception as e:
                    integrity_report['state_files'][state_file.name] = {
                        'status': 'corrupted',
                        'error': str(e)
                    }
            
            # Verify config files
            for config_file in self.config_dir.glob("*.json"):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    integrity_report['config_files'][config_file.name] = {
                        'status': 'valid',
                        'timestamp': data.get('timestamp', 'unknown'),
                        'version': data.get('version', 'unknown')
                    }
                except Exception as e:
                    integrity_report['config_files'][config_file.name] = {
                        'status': 'corrupted',
                        'error': str(e)
                    }
            
            # Verify history files
            for history_file in self.history_dir.glob("*.json"):
                try:
                    with open(history_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    integrity_report['history_files'][history_file.name] = {
                        'status': 'valid',
                        'entry_count': len(data),
                        'latest_entry': data[-1].get('timestamp', 'unknown') if data else 'none'
                    }
                except Exception as e:
                    integrity_report['history_files'][history_file.name] = {
                        'status': 'corrupted',
                        'error': str(e)
                    }
            
            # Determine overall status
            all_files = (list(integrity_report['state_files'].values()) +
                        list(integrity_report['config_files'].values()) +
                        list(integrity_report['history_files'].values()))
            
            if all(f['status'] == 'valid' for f in all_files):
                integrity_report['overall_status'] = 'healthy'
            elif any(f['status'] == 'corrupted' for f in all_files):
                integrity_report['overall_status'] = 'corrupted'
            else:
                integrity_report['overall_status'] = 'warning'
            
            return integrity_report
            
        except Exception as e:
            logger.error(f"❌ Integrity check failed: {e}")
            return {'error': str(e)}


# Global persistence manager instance
persistence_manager = DataPersistence()


def get_persistence_manager() -> DataPersistence:
    """Get the global persistence manager instance."""
    return persistence_manager