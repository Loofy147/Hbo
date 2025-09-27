# quick_start.py - مثال رياضي بسيط لاختبار النظام

def mathematical_objective(trial):
    """
    دالة الهدف الرياضية التي سيتم تحسينها.
    متوافقة مع كائن التجربة (TrialObject) من النظام الجديد.

    الهدف: تعظيم f(x, y) = -(x - 3)^2 - (y + 2)^2 + 10
    الحل الأمثل هو عند x=3, y=-2.

    Args:
        trial (TrialObject): كائن التجربة من نظام HPO.

    Returns:
        float: قيمة الدالة عند النقطة الحالية.
    """
    # النظام الجديد يستخدم trial.suggest_float مباشرة
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -10, 10)

    # حساب قيمة الدالة
    value = -(x - 3)**2 - (y + 2)**2 + 10

    return value