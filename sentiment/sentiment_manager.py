"""
Gestión de Sentimiento del Sistema HRM

Este módulo maneja la descarga, procesamiento y análisis de datos de sentimiento,
incluyendo integración con BERT y gestión de caché de sentimiento.
"""

import asyncio
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timezone
import json
import os

from core.logging import logger
from l3_strategy.sentiment_inference import (
    download_reddit, 
    download_news, 
    infer_sentiment, 
    _load_sentiment_bert_cache,
    get_cached_sentiment_score
)


class SentimentManager:
    """Gestor centralizado de sentimiento para el sistema HRM."""
    
    def __init__(self, cache_update_interval: int = 2160):
        """
        Inicializa el gestor de sentimiento.
        
        Args:
            cache_update_interval: Intervalo de actualización del cache en ciclos (default: 2160 ~ 6 horas)
        """
        self.cache_update_interval = cache_update_interval
        self.last_sentiment_update = 0
        self.sentiment_texts_cache = []
        self.sentiment_score_cache = None
        self.sentiment_timestamp = None
        
    async def update_sentiment_texts(self) -> List[str]:
        """
        Actualiza los textos de sentimiento desde Reddit y News API.
        
        Returns:
            List[str]: Lista de textos para análisis de sentimiento
        """
        texts_list = []
        
        try:
            logger.info("🔄 SENTIMENT: Iniciando descarga de datos de sentimiento...")
            
            # Descargar datos de Reddit
            df_reddit = await download_reddit()
            logger.info(f"✅ SENTIMENT: Descargados {len(df_reddit)} posts de Reddit")

            # Descargar datos de noticias
            df_news = download_news()
            logger.info(f"✅ SENTIMENT: Descargados {len(df_news)} artículos de noticias")

            # Combinar todos los textos
            df_all = pd.concat([df_reddit, df_news], ignore_index=True)

            # Validar DataFrame
            if df_all.empty or 'text' not in df_all.columns:
                logger.warning("⚠️ SENTIMENT: DataFrames vacíos o sin columna 'text', no se puede procesar")
                return []

            df_all.dropna(subset=['text'], inplace=True)

            if df_all.empty:
                logger.warning("⚠️ SENTIMENT: No se obtuvieron textos válidos después de limpieza")
                return []

            texts_list = df_all['text'].tolist()
            logger.info(f"📊 SENTIMENT: {len(texts_list)} textos recolectados para análisis - primeros 3: {texts_list[:3] if texts_list else 'None'}")

            # Validar lista de textos
            if not isinstance(texts_list, list):
                logger.error(f"❌ SENTIMENT: texts_list no es una lista: {type(texts_list)}")
                return []

            if len(texts_list) > 0 and not all(isinstance(t, str) for t in texts_list):
                logger.error(f"❌ SENTIMENT: texts_list contiene elementos no string: {[type(t) for t in texts_list[:5]]}")
                return []

            # Realizar inferencia de sentimiento
            sentiment_results = infer_sentiment(texts_list)
            logger.info(f"🧠 SENTIMENT: Análisis completado para {len(sentiment_results)} textos")

            # Actualizar cache
            self.sentiment_texts_cache = texts_list
            self.sentiment_timestamp = datetime.now(timezone.utc)
            
            return texts_list

        except Exception as e:
            logger.error(f"❌ SENTIMENT: Error actualizando datos de sentimiento: {type(e).__name__}: {e}")
            logger.error(f"❌ SENTIMENT: texts_list en error: {len(texts_list) if isinstance(texts_list, list) else 'No definido'}")
            return []

    def get_sentiment_score(self, max_age_hours: float = 6.0) -> Optional[float]:
        """
        Obtiene el score de sentimiento desde cache o realiza análisis si es necesario.
        
        Args:
            max_age_hours: Edad máxima del cache en horas (default: 6 horas)
            
        Returns:
            Optional[float]: Score de sentimiento o None si no disponible
        """
        # Intentar obtener desde cache BERT
        sentiment_score = get_cached_sentiment_score(max_age_hours=max_age_hours)
        
        if sentiment_score is not None:
            self.sentiment_score_cache = sentiment_score
            logger.debug(f"✅ SENTIMENT: Score obtenido desde cache BERT: {sentiment_score:.4f}")
            return sentiment_score
        
        # Si no hay cache BERT, usar cache interno
        if self.sentiment_score_cache is not None and self.sentiment_timestamp:
            cache_age_hours = (datetime.now(timezone.utc) - self.sentiment_timestamp).total_seconds() / 3600
            if cache_age_hours <= max_age_hours:
                logger.debug(f"✅ SENTIMENT: Score obtenido desde cache interno: {self.sentiment_score_cache:.4f}")
                return self.sentiment_score_cache
        
        # Si no hay cache válido, intentar analizar textos en cache
        if self.sentiment_texts_cache:
            try:
                from l3_strategy.sentiment_inference import predict_sentiment, load_sentiment_model
                tokenizer, sentiment_model = load_sentiment_model()
                sentiment_score = predict_sentiment(self.sentiment_texts_cache, tokenizer, sentiment_model)
                self.sentiment_score_cache = sentiment_score
                self.sentiment_timestamp = datetime.now(timezone.utc)
                logger.info(f"🧠 SENTIMENT: Score calculado desde textos en cache: {sentiment_score:.4f}")
                return sentiment_score
            except Exception as e:
                logger.error(f"❌ SENTIMENT: Error calculando sentimiento desde cache: {e}")
        
        logger.warning("⚠️ SENTIMENT: No hay datos de sentimiento disponibles")
        return None

    async def should_update_sentiment(self, cycle_id: int) -> bool:
        """
        Determina si se debe actualizar el sentimiento basado en el intervalo de cache.
        
        Args:
            cycle_id: ID del ciclo actual
            
        Returns:
            bool: True si se debe actualizar el sentimiento
        """
        # Calcular ciclos desde la última actualización
        cycles_since_last_update = max(0, cycle_id - self.last_sentiment_update)
        cache_expired = _load_sentiment_bert_cache() is None

        if cache_expired and cycles_since_last_update >= self.cache_update_interval:
            logger.info(f"🔄 SENTIMENT: Cache expirado y {cycles_since_last_update} >= {self.cache_update_interval} ciclos - Actualizando datos frescos")
            return True
        elif cache_expired:
            logger.debug(f"⏳ SENTIMENT: Cache expirado pero {cycles_since_last_update} < {self.cache_update_interval} ciclos - Actualización bloqueada por cooldown")
            return False
        else:
            logger.debug(f"✅ SENTIMENT: Cache válido - usando datos en caché (cycle {cycle_id})")
            return False

    async def get_fresh_sentiment_data(self, cycle_id: int) -> Dict:
        """
        Obtiene datos de sentimiento frescos o en caché.
        
        Args:
            cycle_id: ID del ciclo actual
            
        Returns:
            Dict con datos de sentimiento
        """
        # Verificar si se debe actualizar
        should_update = await self.should_update_sentiment(cycle_id)
        
        if should_update:
            # Actualizar textos de sentimiento
            sentiment_texts = await self.update_sentiment_texts()
            self.last_sentiment_update = cycle_id
            
            # Obtener score de sentimiento
            sentiment_score = self.get_sentiment_score()
            
            return {
                'texts': sentiment_texts,
                'score': sentiment_score,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'source': 'fresh_update',
                'cycle_id': cycle_id
            }
        else:
            # Usar datos en caché
            sentiment_score = self.get_sentiment_score()
            
            return {
                'texts': self.sentiment_texts_cache,
                'score': sentiment_score,
                'timestamp': self.sentiment_timestamp.isoformat() if self.sentiment_timestamp else None,
                'source': 'cache',
                'cycle_id': cycle_id
            }

    def get_sentiment_summary(self) -> Dict:
        """
        Obtiene un resumen del estado actual del sentimiento.
        
        Returns:
            Dict con resumen del sentimiento
        """
        return {
            'texts_count': len(self.sentiment_texts_cache),
            'score': self.sentiment_score_cache,
            'timestamp': self.sentiment_timestamp.isoformat() if self.sentiment_timestamp else None,
            'last_update_cycle': self.last_sentiment_update,
            'cache_age_hours': (datetime.now(timezone.utc) - self.sentiment_timestamp).total_seconds() / 3600 
                if self.sentiment_timestamp else None
        }

    def save_sentiment_state(self, filepath: str = "sentiment_state.json"):
        """
        Guarda el estado del sentimiento en archivo.
        
        Args:
            filepath: Ruta del archivo de guardado
        """
        try:
            state = {
                'sentiment_texts_cache': self.sentiment_texts_cache,
                'sentiment_score_cache': self.sentiment_score_cache,
                'sentiment_timestamp': self.sentiment_timestamp.isoformat() if self.sentiment_timestamp else None,
                'last_sentiment_update': self.last_sentiment_update,
                'cache_update_interval': self.cache_update_interval
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.info(f"💾 SENTIMENT: Estado guardado en {filepath}")
            
        except Exception as e:
            logger.error(f"❌ SENTIMENT: Error guardando estado: {e}")

    def load_sentiment_state(self, filepath: str = "sentiment_state.json"):
        """
        Carga el estado del sentimiento desde archivo.
        
        Args:
            filepath: Ruta del archivo de carga
        """
        try:
            if not os.path.exists(filepath):
                logger.info(f"📄 SENTIMENT: No existe archivo de estado, iniciando con estado vacío")
                return

            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.sentiment_texts_cache = state.get('sentiment_texts_cache', [])
            self.sentiment_score_cache = state.get('sentiment_score_cache')
            self.last_sentiment_update = state.get('last_sentiment_update', 0)
            self.cache_update_interval = state.get('cache_update_interval', 2160)
            
            if state.get('sentiment_timestamp'):
                self.sentiment_timestamp = datetime.fromisoformat(state['sentiment_timestamp'])
            
            logger.info(f"📂 SENTIMENT: Estado cargado desde {filepath}")
            
        except Exception as e:
            logger.error(f"❌ SENTIMENT: Error cargando estado: {e}")


# Funciones de conveniencia para compatibilidad con main.py existente

async def update_sentiment_texts() -> List[str]:
    """
    Función de conveniencia para actualizar textos de sentimiento.
    Mantiene compatibilidad con el código existente en main.py.
    
    Returns:
        List[str]: Lista de textos para análisis de sentimiento
    """
    manager = SentimentManager()
    return await manager.update_sentiment_texts()


def get_sentiment_score(max_age_hours: float = 6.0) -> Optional[float]:
    """
    Función de conveniencia para obtener score de sentimiento.
    Mantiene compatibilidad con el código existente en main.py.
    
    Args:
        max_age_hours: Edad máxima del cache en horas
        
    Returns:
        Optional[float]: Score de sentimiento o None si no disponible
    """
    manager = SentimentManager()
    return manager.get_sentiment_score(max_age_hours)