# Comparación observacional — primera versión, 27 octubre 2026

## Alcance físico

`src/observations` reutiliza `rho_of_a`, `hubble`, `D` y los parámetros de
`src.models`. Soporta `BarotropicPowerDiffusionModel`, `PhantomQ0Model` y
`PhantomQAlphaRhoModel`. No copia sus ecuaciones diferenciales ni interpreta
automáticamente otros modelos. No modifica el core histórico.

Para el fluido único se usa E(z)=H_core[rho(a0/(1+z))]/H_core(rho0).
H(z)=H0 E(z), con H0 en km/s/Mpc. Las densidades del core pueden normalizarse
con rho0=1: alpha representa entonces bar_alpha; A representa bar_A.
La escala física total satisface 3H0²=rho_tot,0 en unidades gravitacionales;
no se identifica el número H0=70 con un H del core sin convertir unidades.

Si se configura omega_m>0 se adopta explícitamente otra hipótesis:

    E²(z)=Omega_m(1+z)^3+Omega_r(1+z)^4
          +(1-Omega_m-Omega_r)*[H_core(rho(z))/H_core(rho0)]².

Esto añade materia/radiación conservadas por separado; Q intercambia energía
sólo con el fluido oscuro. Es una extensión del modelo de fluido único, no un
resultado del core previo. La relación rho(a) sigue siendo válida:
`D*d(rho)/dln(a)=-3(rho+p)`, donde H se cancela. En cambio, dot(H) del sistema
extendido NO coincide con el método hdot del fluido único. Las clasificaciones
futuras del core describen ese sector original; no se reetiquetan automáticamente
como destinos completos de esta extensión.

No se cruza una raíz de D ni se extrapolan distancias fuera del intervalo
calculado. El modelo del rip difusivo puede tener un límite pasado antes del
redshift solicitado: un error de dominio significa incompatibilidad con ese
intervalo bajo el ansatz, no un fallo que deba ocultarse con extrapolación.

## Observables y convenciones

Universo plano, c=299792.458 km/s:

    DM=(c/H0)*integral_0^z dz/E(z), DH=c/H(z),
    DV=[z*DM²*DH]^(1/3), DA=DM/(1+z),
    DL=(1+zHEL)*DM(zHD) para las supernovas,
    mu=5 log10(DL/Mpc)+25.

Se integra sobre una malla log(1+z), Simpson acumulado e interpolación PCHIP.
El ejemplo repite chi² con 4097 y 8193 nodos. El adaptador permite cambiar
la resolución. No extrapolar este cálculo de fondo tardío a recombinación.

BAO utiliza DM/rd, DH/rd y DV/rd con rd en Mpc. **rd es libre**: no se calcula
el horizonte acústico con un modelo de fluido phantom tardío. SN sin calibración
y BAO restringen H0*rd y la forma de E(z); no separan H0 y rd. En el ejemplo
H0=70 es una referencia de escala, no una medida ni un prior cosmológico.

SN: se ajusta m_b_corr=mu+M y se perfila analíticamente el intercepto M usando
la covarianza completa: M=(1ᵀ C^-1 r)/(1ᵀ C^-1 1), r=m_b_corr-mu.
No se utiliza MU_SH0ES y no se añade un prior SH0ES implícito. El perfil no es
una marginalización bayesiana con prior propio. La normalización gaussiana está
disponible en `Gaussian.loglike`; `joint_chi2` devuelve chi² y M perfilada.
La combinación supone independencia entre los dos conjuntos; no combina
catálogos de supernovas que se solapen ni DESI DR1 con DR2.

## Datos descargados y fuentes

- DESI DR2 BAO, 13 entradas: medias y covarianza de `ALL_GCcomb`.
  La [página oficial DESI](https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/README.html)
  enlaza el [repositorio de likelihoods](https://github.com/CobayaSampler/bao_data/tree/master/desi_bao_dr2).
  Referencia: [DESI DR2 Results II](https://arxiv.org/abs/2503.14738).
  No se han añadido los productos Alcock–Paczynski DR2 de 2026: requieren
  revisar selección, correlaciones y posible solapamiento antes de combinarlos.
- [Pantheon+ publicación oficial de datos](https://github.com/PantheonPlusSH0ES/DataRelease/tree/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR):
  Pantheon+SH0ES.dat, STAT+SYS.cov y README. 1701 entradas originales.
  Selección explícita `zHD>0.01 & IS_CALIBRATOR==0`: 1580 entradas, no
  necesariamente 1580 supernovas únicas. Se mantienen las mediciones repetidas
  y sus correlaciones. Este corte conservador es nuestra elección, no una
  afirmación de reproducir exactamente la selección de cada análisis publicado.
  Referencia indicada por el release: https://arxiv.org/abs/2202.04077.

Las matrices se recortan con exactamente los mismos índices que los vectores.
No se suman de nuevo los errores de la tabla ni se reemplaza la matriz por
su diagonal. Se exige finitud, simetría y factorización de Cholesky positiva.
Las barras de error de los gráficos BAO son sólo visualizaciones diagonales;
los ajustes siempre usan la matriz completa.

En `data/observations/{desi_dr2,pantheon_plus}/manifest.json` se guardan URL
por commit, fecha UTC, bytes y SHA256. Las descargas no sobrescriben una versión
existente distinta. Los archivos locales se pueden usar sin conexión. No se
descargan los catálogos de galaxias completos: los productos BAO comprimidos
son suficientes para esta verosimilitud de fondo, bajo sus hipótesis estándar.

## Ejecución

Desde `cosmo_singularities/`:

```bash
python -m scripts.download_observations
python -m unittest discover -s tests -v
python -m scripts.compare_observations
```

La descarga usa curl con validación TLS; no requiere paquetes nuevos. Para una
descarga manual, los nombres y URLs originales están en el script y manifiestos.
Cada comparación crea una carpeta nueva en `data/observations/results/`.

El ejemplo compara LCDM (límite w=-1,alpha=0 del propio core), phantom sin
difusión (w=-1.2) y phantom con difusión (w=-1.2,bar_alpha=4,beta=-1), todos
con materia conservada. Ajusta Omega_m y H0*rd, y perfila M. Usa dos inicios
con límites explícitos, registra convergencia y cercanía a los límites, y
comprueba resolución numérica. **Los parámetros del fluido están FIJOS**;
no interpretar esta comparación como posterior, límite de confianza o evidencia
de preferencia por una familia completa. El lector `fit_background` acepta
una factoría para variar otros parámetros, pero requiere revisar degeneraciones
y dominio antes de construir un barrido multidimensional.

## Pruebas y próximos pasos

Pruebas: límite LCDM con cuadratura independiente, de Sitter, recuperación del
fluido único, dualidad de distancias, corrección heliocéntrica, covarianza
correlacionada, invariancia SN frente a H0 al perfilar M, degeneración H0*rd,
rechazo de frontera pasada y recuperación de parámetros en datos sintéticos.

Pendiente para inferencia de publicación: variar bar_alpha,beta,omega o A,nu;
evaluar identificabilidad, priors y selección; perfiles/intervalos o muestreo
posterior con diagnóstico de convergencia; estudiar hipótesis termodinámicas
por separado; auditar el tratamiento de sistemáticos y posibles correlaciones.
Cronómetros cósmicos pueden incorporarse después con covarianzas y fuentes
homogéneas. CMB/full-shape requieren una descripción del universo temprano y
perturbaciones: no se simulan con este módulo de distancias.
