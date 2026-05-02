# Resumen de Decisiones — Challenge Data Scientist Entel
## Predicción de Churn TELCO

---

## Decisiones más relevantes

### EDA y preparación de datos

| # | Decisión | Justificación |
|---|----------|---------------|
| 1 | Split temporal (Jun+Jul train / Ago test) | Simula escenario productivo. Split aleatorio genera data leakage |
| 2 | Registros NaT descartados (1.701 / 0.57%) | 100% de variables de comportamiento nulas. Sin información útil |
| 3 | Imputación por semántica de negocio (fillna=0) | Nulo = inactividad verificada (total_og/ic_mou = 0). No es error |
| 4 | Segmentación ARPU + Clustering RFM | Nueva variable de segmento como feature predictora del modelo |

### Modelamiento

| # | Decisión | Justificación |
|---|----------|---------------|
| 5 | Métrica principal: P@2000 | Restricción fija de 2.000 contactos. El ROI depende del top, no del ranking global |
| 6 | Optimización por Recall | FN cuesta 3x más que FP ($3.000 vs $1.000). Minimizar FN maximiza ROI |
| 7 | Random Forest ratio=0.10 tuneado por Recall | Único modelo con ROI positivo (+$235.000 CLP) |
| 8 | Priorización por ranking de score descendente | Garantiza exactamente 2.000 contactos priorizando mayor riesgo |

---

## Supuestos clave

| Supuesto | Valor |
|----------|-------|
| Contactos / Costo / Valor retenido / Tasa éxito | 2.000 / $1.000 / $30.000 / 10% sobre churners reales |
| Break-even P@2000 | 33.33% → modelo opera con margen de +4 pts (37.25%) |
| Churn | Inactividad mensual reversible (42.1% de churners de Jun reaparecen en Jul) |
| N óptimo sin restricción | 1.400 clientes → ROI +$277.000 CLP |

---

## Resultados del modelo final

| P@2000 | Churners top 2k | Lift | ROI |
|--------|-----------------|------|-----|
| 37.25% | 745 / 1.882 (39.6%) | 19.6x | **+$235.000 CLP** |

---

## Principales riesgos

| # | Riesgo | Mitigación |
|---|--------|------------|
| 1 | Solo 2 meses de historia | Incorporar Mayo cuando esté disponible |
| 2 | Churn creciente (1.04% → 1.90%) → degradación del modelo | Reentrenamiento trimestral o cuando P@2000 < 33.33% |
| 3 | Desbalance extremo (1:77) → sensible a cambios en producción | Monitoreo mensual de distribución de features |
| 4 | Margen de seguridad de solo ~4 puntos sobre break-even | Monitoreo mensual de P@2000 |

---

## Mejoras propuestas

1. **Mayo disponible** → deltas completos + aceleración de ARPU → estimación +3-5 pts P@2000
2. **Custom scorer P@2000** en tuning en lugar de Recall
3. **LightGBM** → mejor rendimiento en datasets con desbalance extremo
4. **Producción:** top 15 variables + monitoreo mensual + reentrenamiento por umbral 33.33%
5. **Medios propios** (SMS, email) para clientes de propensión media → ampliar alcance sin costo de $1.000

## Adicional
Se construyeron variables de cambio entre meses (delta_arpu,
delta_total_og_mou, etc.) pero su incorporación al modelo
deterioró el P@2000 de ~35% a ~27%.

Causa: las 99.999 filas de Junio tienen delta=NaN al no tener
mes anterior. Imputar con 0 introduce ruido porque el modelo
interpreta "sin cambio" cuando en realidad "sin información".

Con Mayo disponible los deltas serían calculables sin NaN
para todos los clientes. 