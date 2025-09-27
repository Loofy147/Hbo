# quick_start.py - مثال رياضي بسيط لاختبار النظام

def mathematical_example(trial):
    """
    دالة الهدف الرياضية التي سيتم تحسينها.
    الهدف هو تعظيم هذه الدالة.
    f(x, y) = -(x - 3)^2 - (y + 2)^2 + 10
    الحل الأمثل هو عند x=3, y=-2 حيث القيمة القصوى هي 10.

    Args:
        trial (optuna.trial.Trial): كائن التجربة من Optuna.

    Returns:
        float: قيمة الدالة عند النقطة الحالية.
    """
    # Optuna يقترح القيم للمعاملات
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)

    # حساب قيمة الدالة
    value = -(x - 3)**2 - (y + 2)**2 + 10

    return value