# Inferencia barotrópica: etapa exploratoria

Fecha: 2026-10-27. Esta etapa valida el fondo y permite omega libre;
no constituye todavía un análisis posterior completo ni resultados finales del paper.

## Hipótesis y normalización

Materia conservada, geometría plana, radiación omitida en el ajuste tardío.
El fluido oscuro conserva p=omega*rho y Q=alpha*rho^beta del manuscrito.
Con x=rho/rho0 y alpha_bar=alpha*rho0^(beta-1):

    dx/d ln a = -3(1+omega)x / (1+beta*alpha_bar*x^(beta-1))
    E² = Om*(1+z)^3 + (1-Om)*(x+alpha_bar*x^beta)/(1+alpha_bar)

La relación rho(a) es la del core: el H total cancela al cambiar la variable
independiente de t a ln(a). Las trayectorias temporales de un solo fluido
no se sustituyen directamente en la cosmología con materia.

Para las reconstrucciones se usa c=(1-Om)/(rho0+Q0):

    Q/(3H0²) = c Q_core
    Qdot/(3H0³) = -3 E c (1+omega) rho (D-1)/D
    Hdot/H0² = -3/2 [Om*(1+z)^3 + c(1+omega)rho]

El adaptador admite además radiación y añade su término 4*Or*(1+z)^4/3
al corchete. H0 dimensional debe convertirse a una unidad de tiempo coherente
si se reconstruyen cantidades dimensionales; los archivos usan razones sin dimensiones.

Se exige rho>0, F>0 y D con el signo inicial en todo el rango de distancias.
No se aplica filtro entrópico: para T>0 su signo depende de
(1+omega)*alpha*beta/D, no sólo de la desigualdad del sector phantom.
En ramas regulares w_DE+1=(1+omega)/F no cambia de signo.

## Identificabilidad exacta

- omega=-1: rho constante, fondo LCDM, difusión no identificable por expansión.
- alpha_bar=0: beta no identificable.
- beta=1: sólo w_eff=(omega-alpha_bar)/(1+alpha_bar) distingue el fondo oscuro.
- beta=0: Q constante, Qdot=0. Se corrigió la omisión de esta solución en
  rho_of_a_lambert del core. Ahora devuelve rho0*(a/a0)^[-3(1+omega)].
- SN sin calibración: H0 y M degenerados. Se perfila M y se fija H0=70 como referencia.
- BAO sin cosmología temprana: se ajusta H0*rd, no H0 y rd por separado.

No se cuenta un parámetro no identificable como una medición. La función
`degeneracies` reconoce identidades exactas, no diagnostica por sí sola
casi-degeneraciones; éstas requieren perfiles y distribuciones posteriores.

## Ejecución

Desde cosmo_singularities:

    OPENBLAS_NUM_THREADS=1 python3 -m scripts.fit_barotropic_observations --dataset joint
    OPENBLAS_NUM_THREADS=1 python3 -m scripts.fit_barotropic_observations --dataset bao
    OPENBLAS_NUM_THREADS=1 python3 -m scripts.fit_barotropic_observations --dataset sn

Cada ejecución crea un directorio nuevo free_omega_* con fits.json,
trayectorias CSV y reconstrucciones PNG. Se guardan hashes del código,
versiones, manifiestos de datos, límites, inicios, convergencia y diferencia
al duplicar la resolución de las distancias.

Los límites exploratorios son Om=[0.05,0.6], omega=[-1.8,-0.3],
alpha_bar=[-0.8,5], beta=[-3,3], H0*rd=[7000,13000] km/s.
Son límites computacionales, NO priors físicamente justificados ni
intervalos observacionales. Se usan dos inicios por modelo y Powell;
no garantizan un mínimo global. Las ramas D<0 inicializadas aquí tienen
alpha_bar>0,beta<0. La componente alpha_bar<0,beta>1 requiere un estudio
separado y no se considera cubierta por esta ejecución.
Los resultados cerca de un límite deben motivar ampliar o reparametrizar
el dominio, no anunciar una detección. Tampoco una reducción de chi² es
por sí sola evidencia de difusión.

## Datos y pendientes

Se usa la covarianza STAT+SYS completa de Pantheon+, m_b_corr, zHD y zHEL;
la selección local es zHD>0.01 e IS_CALIBRATOR=0. Esta selección es explícita,
pero todavía debe contrastarse fila a fila con el likelihood oficial elegido
antes de reproducir restricciones publicadas. No se utiliza MU_SH0ES.
Documentación oficial revisada:
https://github.com/PantheonPlusSH0ES/DataRelease/tree/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR

DESI: se mantiene el vector BAO DR2 y covarianza con procedencia y hash
registrados en su manifiesto. No se incluyen CMB, AP ni full-shape.

Pendientes para conclusiones científicas: auditoría completa contra likelihoods
oficiales, ampliar exploración de ramas/inicios y límites, recuperación de
parámetros con simulaciones, modelos de comparación con cruce, posterior con
convergencia y sensibilidad a priors, y comparación de datos por separado.
No afirmar cruce ni evidencia bayesiana a partir de estos archivos.

### Control del optimizador incorporado

La primera ejecución detectó un mínimo local D>0 peor que el modelo wCDM
anidado. Se conserva con ADVERTENCIA.md y no debe usarse para inferencia.
El ajuste ahora escala cada parámetro al intervalo [0,1], compara Powell
y SLSQP, añade la solución wCDM como semilla exacta en D>0 y rechaza
resultados que empeoren la mejor semilla. La convergencia local sigue sin
certificar un mínimo global. Cada intento registra método y mensaje.

### Precisión cerca de D=0

La inspección de Qdot detectó pérdida de precisión de scipy.special.lambertw
sobre W_-1 muy cerca de -1/e. Se añadió la expansión local de W en
p=±sqrt[2(1+e*z)], hasta orden p^5, evaluando la distancia al punto de rama
mediante log1p/expm1 y los parámetros iniciales. Se impone además la identidad
rho(a0)=rho0. Una prueba compara con la inversión independiente por brentq
de la relación implícita N(rho) y comprueba la transferencia inicial.
Esto no regulariza D=0: su divergencia física sigue presente.
