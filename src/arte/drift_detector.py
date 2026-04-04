from river import drift


# =============================================================================
# ADWINChangeDetector — wrapper que replica o ADWINChangeDetector.java do MOA
# =============================================================================
class ADWINChangeDetector:
    """
    Wrapper sobre river.drift.ADWIN que replica o comportamento do
    ADWINChangeDetector.java do MOA: só sinaliza drift quando o erro
    **aumentou** em relação à estimativa anterior.

    O river.drift.ADWIN dispara drift_detected em qualquer direção
    (melhora ou piora). Isso causa um cascade pós-reset no ARTE: a árvore
    recém-criada aprende rapidamente, o erro cai, e o ADWIN interpreta
    essa melhora como drift → novo reset → espiral.

    O MOA evita isso com a checagem (linha 51 do ADWINChangeDetector.java):
        if (adwin.getEstimation() > ErrEstim)  // só dispara se piorou

    Parâmetros
    ----------
    delta : float
        Parâmetro de confiança do ADWIN (padrão 1e-3, igual ao MOA).
    kwargs :
        Demais parâmetros repassados ao river.drift.ADWIN.
    """

    def __init__(self, delta: float = 0.001, **kwargs):
        self._adwin = drift.ADWIN(delta=delta, **kwargs)
        self._drift_detected = False

    def update(self, value: float):
        prev_estimation = self._adwin.estimation
        self._adwin.update(value)
        # Só sinaliza se o erro aumentou — equivalente ao MOA
        self._drift_detected = (
            self._adwin.drift_detected and
            self._adwin.estimation > prev_estimation
        )

    @property
    def drift_detected(self) -> bool:
        return self._drift_detected

    @property
    def estimation(self) -> float:
        return self._adwin.estimation

    def clone(self):
        return ADWINChangeDetector(
            delta=self._adwin.delta,
            min_window_length=self._adwin.min_window_length,
        )
