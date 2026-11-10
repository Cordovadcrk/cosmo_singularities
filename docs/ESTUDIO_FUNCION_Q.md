# Viabilidad y elección de la función Q

Fecha local: 9 de noviembre de 2026. Los nombres de ejecuciones usan UTC.

## Pregunta y alcance

Se contrasta la ley de potencia con ausencia de Q, Q constante y Q lineal.
Se añade CPL como referencia fenomenológica de energía oscura dinámica que
puede cruzar −1, NO como una solución del modelo unimodular del paper.
Este análisis usa perfiles de verosimilitud: no son distribuciones posteriores,
no se calcula evidencia bayesiana y no se atribuyen significancias de cruce.
No se han comparado todas las posibles funciones no lineales Q(rho).

El fondo es plano, con materia conservada y el sector oscuro barotrópico
interactuando sólo con Q. Se omite radiación en estos ajustes tardíos.
Se mantiene rho>0, rho+Q>0, una rama regular de D y omega constante por
trayectoria. Exigir rho+Q>0 es una hipótesis sobre el sector oscuro: con materia,
la realidad de H total por sí sola sería una condición más débil.
No se impone la segunda ley como prior ni se restringe a priori a phantom.

## Derivación y dimensiones

Con x=rho/rho0, alpha_bar=alpha*rho0^(beta−1) y a0=1:

    D(x) dx/d ln(a) = −3(1+omega)x
    D(x) = 1+alpha_bar*beta*x^(beta−1)
    E² = Om*(1+z)^3 + (1−Om)*(x+alpha_bar*x^beta)/(1+alpha_bar)

La continuidad es la del core; H cancela al usar ln(a). El adaptador reutiliza
rho_of_a, F, D y signed_hubble del core. La vectorización evita miles de
llamadas escalares sin cambiar las ecuaciones. No se integra una copia de la
cosmología para producir los ajustes.

En unidades 8piG=1, rho0=3H0²(1−Om)/(1+alpha_bar). La escala dimensional de
alpha depende de H0, que estos datos sin calibración no determinan por sí solo.
Se reportan alpha_bar y Q/(3H0²), no una medición independiente de alpha físico.

## Datos y auditoría

DESI DR2: 13 observables, vector y covarianza completos oficiales. Los hashes
se verifican antes de cada ejecución. Los nombres de archivos y rs_fid=1
coinciden con el likelihood de referencia:
https://raw.githubusercontent.com/CobayaSampler/cobaya/master/cobaya/likelihoods/bao/desi_dr2/desi_bao_all.yaml

Pantheon+: m_b_corr, zHD, zHEL y matriz STAT+SYS. El corte de referencia es
zHD>0.01 (1590 entradas, no 1590 supernovas únicas). Las repeticiones conservan
sus covarianzas. El código de referencia no excluye IS_CALIBRATOR=1 en este
corte; utilizar sus magnitudes no utiliza las distancias de Cefeidas.
No se usa MU_SH0ES, CEPH_DIST ni una calibración SH0ES.
Referencia consultada y copia local en docs/observational_sources:
https://raw.githubusercontent.com/CobayaSampler/cobaya/master/cobaya/likelihoods/sn/pantheonplus.py

El lector conserva su selección histórica por defecto para no cambiar
silenciosamente notebooks previos. Este estudio solicita explícitamente
selection='cosmology'. La selección anterior (1580 entradas) se evalúa como
sensibilidad. No deben compararse chi² absolutos entre las dos muestras como
si representaran una mejora de modelo.

Se verifica el perfil de magnitud frente a la forma de inversa proyectada
usada en el likelihood de referencia. Es una validación algebraica/numérica,
no una reproducción de todas las cadenas publicadas, que pueden usar otros
conjuntos de datos y priors.

## Parámetros de calibración perfilados

Para r=m−mu y C la covarianza, el intercepto óptimo es
M=(1^T C^−1 r)/(1^T C^−1 1), con chi²=(r−M1)^T C^−1(r−M1).
H0=70 km/s/Mpc se fija como referencia de escala, no como medición o prior.

Para BAO la predicción es u/S con S=H0*rd>0 y u evaluado en S=1 km/s.
Con L L^T=C, u_w=L^−1u e y_w=L^−1y:

    1/S = (u_w^T y_w)/(u_w^T u_w).

Así se perfila S exactamente y no se pierde tiempo explorando una degeneración
artificial entre H0 y rd. El resultado se valida frente a minimización numérica
independiente. Si no hay BAO, S no se incluye. Perfilar S no equivale a
marginalizarlo con un prior bayesiano.

## Modelos y equivalencias

- LCDM: expansión de referencia, omega=−1.
- noQ: alpha_bar=0 y omega libre.
- Q constante: beta=0, alpha_bar y omega libres; Qdot=0.
- Q lineal: beta=1. El fondo sólo depende de
  w_eff=(omega−alpha_bar)/(1+alpha_bar). Es idéntico a noQ con ese w_eff.
  Se compara el mismo dominio de w_eff, no cajas diferentes de omega y alpha.
- Potencia general: omega, alpha_bar, beta y Om libres, con ramas D>0,
  D<0/alpha_bar>0/beta<0 y D<0/alpha_bar<0/beta>1 explícitas.
- CPL: w(z)=w0+wa*z/(1+z), con distancias y likelihood compartidos.
  Integrar la continuidad conservada da
  epsilon(z)/epsilon0=(1+z)^[3(1+w0+wa)] exp[−3wa*z/(1+z)].
  Cuando wa=0 coincide numéricamente con noQ. Referencias:
  https://arxiv.org/abs/gr-qc/0009008 (IJMPD 10, 213, 2001)
  https://arxiv.org/abs/astro-ph/0208512 (PRL 90, 091301, 2003).

Una ley afín Q=q_c+lambda*rho tampoco identifica lambda con datos de fondo
solamente: con r=(1+lambda)rho, epsilon=r+q_c y
p_eff=[(omega−lambda)/(1+lambda)]r−q_c. Para 1+lambda>0 equivale al caso
Q constante con omega redefinido. Es una identidad analítica, no un ajuste nuevo.

## Búsqueda y sensibilidad

Se usa differential_evolution con semilla registrada y refinamientos SLSQP y
Powell, en coordenadas escaladas a [0,1]. Se conservan las semillas válidas
si el optimizador empeora el objetivo. Se registra la convergencia local y
los fallos; no se certifica un mínimo global.

La primera búsqueda usa Om=[0.05,0.6], omega=[−1.8,−0.3],
alpha_bar=[−0.8,5], beta=[−3,3]. La ampliación usa Om=[0.01,0.8],
omega=[−3,0.5], alpha_bar=[−0.95,20], beta=[−6,6]. No son priors posteriores.
La ampliación permite incluso presión material positiva: lo que se contrasta
observacionalmente es el sector oscuro efectivo. La subfamilia del paper
omega<−1 se ajusta por separado, con borde superior −1.000001; éste aproxima
un límite, no demuestra un máximo interior phantom.

Se impone inicialmente |D0|>=1e−5 y se compara con 0.001, 0.01 y 0.1 en
la rama alpha_bar>0,D<0. Son cortes de regularidad/numeración explícitos,
no restricciones observacionales. Se registran los puntos próximos al corte.

Se hacen perfiles discretos de beta, alpha_bar y de omega en noQ. Cada valor
fijo reoptimiza los demás parámetros. El perfil de beta conserva la solución
anidada alpha_bar=0 y explora ambas componentes D<0 cuando existen. Se reparan
búsquedas sin puntos válidos con semillas físicas cercanas a omega=−1; un fallo
de muestreo no se interpreta como exclusión física.

La resolución principal de distancias es 1025, y cada resultado se contrasta
con 4097 y 8193 puntos. Se exige cambio en chi² menor que 0.01. Sólo se interpretan
puntos que pasan este control. Además se contrasta Lambert con integración
independiente en ln(a) para los resultados finales.

## Qué significa justificar Q

Viabilidad observacional no selecciona una función de forma única. En una
historia epsilon(z) conocida, para omega constante distinto de −1:

    rho(z) = (1+z) epsilon'(z) / [3(1+omega)]
    Q(z) = epsilon(z) − rho(z).

Esto se obtiene directamente de epsilon_dot=−3H(1+omega)rho y
z_dot=−(1+z)H. Cuando rho(z) es invertible se construye Q(rho). Distintos
omega pueden cambiar esa descomposición, respetando rho>0 sólo donde
los signos lo permitan. Es una explicación analítica de por qué una buena
expansión no determina por sí sola la ley constitutiva.

En nuestro modelo w_DE+1=(1+omega)/F. El cruce regular es imposible bajo las
hipótesis impuestas. Un cruce en el mejor ajuste CPL no demuestra que los datos
lo exijan y no se transfiere al modelo unimodular.

La familia de potencia tiene parámetros no identificables cuando alpha=0 u
omega=−1, y fronteras singulares. Por ello no se convierten automáticamente
Delta chi² en sigmas, ni se usan penalizaciones nominales AIC/BIC como evidencia
concluyente. Los perfiles discretos tampoco son intervalos posteriores.

## Reproducción

Desde cosmo_singularities, con las versiones de requirements.txt:

    OPENBLAS_NUM_THREADS=1 python3 -m scripts.study_diffusion_evidence
    OPENBLAS_NUM_THREADS=1 python3 -m scripts.refine_Q_constraints RUTA_DEL_RESULTADO
    OPENBLAS_NUM_THREADS=1 python3 -m scripts.report_Q_evidence RUTA_DEL_RESULTADO

La ejecución inicial deja report.json; el refinamiento guarda constraints.json,
sin reemplazar los resultados iniciales. Los hashes de código/datos, intentos,
límites y controles quedan en los archivos. El notebook de entrega lee una
ejecución explícita, para no incorporar accidentalmente una corrida incompleta.

## Límite del trabajo terminado

Esta entrega completa una comparación de perfiles y controles numéricos para
estos datos y dominios. Una afirmación de evidencia bayesiana requeriría priors
propios justificados, muestreo convergente y sensibilidad a esos priors. Una
significancia frecuentista de detección de difusión requeriría calibrar con
simulaciones el estadístico bajo el nulo singular. No se presentan esas
magnitudes como si ya se hubieran calculado.
