from river import metrics
import collections


# ============================================================================
# 8. KAPPA M METRIC
# ============================================================================
class KappaM(metrics.base.MultiClassMetric):
    """
    Kappa M Statistic (Bifet et al., 2015).
    
    Atribui valor zero ao classificador de classe majoritária,
    sendo mais rigoroso que o Kappa padrão.
    """
    def __init__(self):
        super().__init__()
        self._confusion_matrix = collections.defaultdict(lambda: collections.defaultdict(int))
        self._class_counts = collections.Counter()
        self._total = 0
    
    def update(self, y_true, y_pred, sample_weight=1.0):
        self._confusion_matrix[y_true][y_pred] += sample_weight
        self._class_counts[y_true] += sample_weight
        self._total += sample_weight
    
    def get(self):
        if self._total == 0:
            return 0.0
        
        # Acurácia observada
        correct = sum(
            self._confusion_matrix[cls][cls] 
            for cls in self._class_counts
        )
        p0 = correct / self._total
        
        # Classe majoritária
        majority_class = max(self._class_counts, key=self._class_counts.get)
        majority_count = self._class_counts[majority_class]
        
        # Acurácia esperada do majority classifier
        pc = majority_count / self._total
        
        # Kappa M
        if pc >= 1.0:
            return 0.0
        
        kappa_m = (p0 - pc) / (1.0 - pc)
        return kappa_m
    
    def revert(self, y_true, y_pred, sample_weight=1.0):
        self._confusion_matrix[y_true][y_pred] -= sample_weight
        self._class_counts[y_true] -= sample_weight
        self._total -= sample_weight

