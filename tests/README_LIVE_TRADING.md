# HRM Live Trading Setup & Fee Monitoring Guide

---

## ⚠️ ADVERTENCIA CRÍTICA - DOCUMENTO HISTÓRICO

> **ESTE DOCUMENTO ES HISTÓRICO - CONFIGURACIÓN DE REFERENCIA ÚNICAMENTE**
> 
> El sistema **actualmente opera con PAPER_MODE=True forzado** en `core/config.py` y `main.py`.
> 
> **Para activar live trading:**
> 1. Establecer explícitamente `PAPER_MODE=false` en variables de entorno
> 2. Revisar la sección "Modos de Operación" en `readme.md`
> 3. Confirmar manualmente con espera de seguridad de 10 segundos
> 
> **⚠️ RIESGO:** Este documento describe configuración LIVE pero el código fuerza PAPER_MODE por seguridad.

---

## 🚀 Live Trading Configuration Completed

The system has been configured for live trading with the following settings:

### Core Configuration
- **BINANCE_MODE**: LIVE (real money trading)
- **USE_TESTNET**: false (live market data and execution)
- **OPERATION_MODE**: LIVE (production trading)
- **PAPER_MODE**: false (real order execution)
- **HRM_PATH_MODE**: PATH3 (full L3 dominance with auto-rebalance - "gallina de los huevos de oro")

### Risk Management (Live Trading Conservative)
- **Portfolio Limits**:
  - BTC Exposure: 40% maximum
  - ETH Exposure: 40% maximum
  - Individual Position: $1,200 maximum (~40% of $3k portfolio)
- **Cash Reserves**: 20% minimum USDT liquidity, $500 absolute minimum
- **Trading Limits**: 10 trades per day maximum, 10% drawdown stop-out

### PATH3 Auto-Rebalance System (The Golden Goose)
Enabled with comprehensive safety features:
- **Circuit Breaker**: ENABLE_AUTO_REBALANCE=true
- **Checksum Verification**: Audit logging for forensic analysis
- **Dry-run Disabled**: Real execution with fees buffer (1.01x for sells)
- **Extended Cooldown**: 5-minute recovery period between rebalances
- **Min-Order Checks**: USDT balance validation for buys

## 💰 Fee Monitoring Considerations for Live Markets

### Binance Trading Fees (Maker/Taker)
- **Spot Trading**: 0.1% (maker) / 0.1% (taker) for default accounts
- **VIP Tiers**: Lower fees (0.09% to 0.036%) based on 30-day trading volume
- **BNB Discount**: 25% reduction when paying fees with BNB

### Fee Impact Calculation
Monthly fee cost estimation for active trading:
```
Daily Volume: $10,000 (conservative estimate)
Daily Fee Cost: $10 (0.1% of $10,000)
Monthly Fee Cost: ~$300
Annual Fee Cost: ~$3,600 (10% of $3k portfolio!)
```

### Real Market Considerations

#### 1. **Price Slippage**
- Live markets have wider spreads than testnet
- High volatility pairs (BTC/ETH) show 0.2-0.5% spreads vs 0.01-0.02% on testnet
- Market orders execute at worse prices than limit orders

#### 2. **Network Congestion**
- High volatility periods cause increased gas fees (though not applicable for spot)
- Market makers may widen spreads during news events
- Order book depth can be reduced during extreme volatility

#### 3. **Trading Frequency Impact**
PATH3 auto-rebalance may trigger frequent rebalancing during market stress:
- 5+ stop-loss triggers → automatic rebalance
- Extended cooldown periods help, but can still cause transaction costs
- Consider disabling auto-rebalance during extreme volatility if fees become excessive

#### 4. **Tax Implications** (Spain)
- Trading profits subject to 19-23% capital gains tax
- Track all trades including fees for accurate reporting
- FIFO/LIFO method affects tax calculation
- Consider tax-loss harvesting opportunities

### Fee Monitoring Dashboard

Monitor these metrics daily:

```
Portfolio Value: $XXXX.XX
Asset Allocation: BTC XX% | ETH XX% | USDT XX%
Trading Fees Paid (Month): $XX.XX
Fee % of Portfolio: X.XX%
Trades This Month: XX
Average Fee per Trade: $X.XX
```

### Optimization Strategies

1. **Reduce Trading Frequency**:
   - Increase stop-loss thresholds if possible
   - Extend rebalance cooldowns during high volatility
   - Use limit orders when possible to get maker fees

2. **Fee Tier Promotion**:
   - Track 30-day volume for VIP tier advancement
   - Pay fees with BNB for 25% discount
   - Consider spot market making strategies (future enhancement)

3. **Performance Attribution**:
   - Compare strategy returns net of fees vs gross
   - Monitor if PATH3 rebalance costs exceed benefits
   - Consider fee-free alternatives during low conviction periods

### Emergency Fee Management

If fees exceed 1% of portfolio monthly:
```
- Disable auto-rebalance: ENABLE_AUTO_REBALANCE=false
- Shift to less aggressive position sizing
- Consider market timing (reduce activity during high fee periods)
- Review strategy profitability net of costs
```

### Live Trading Checklist

- [ ] API keys configured securely (not in .env)
- [ ] Initial capital deposited (verify balance)
- [ ] Test small orders manually first
- [ ] Monitor first few trades for execution quality
- [ ] Set up fee monitoring alerts
- [ ] Configure automated backup withdrawals if needed
- [ ] Document tax basis for positions
- [ ] Set profit-taking thresholds accounting for fees

**Remember**: The PATH3 auto-rebalance system is your "golden goose" but must be monitored for fee efficiency in live markets.
