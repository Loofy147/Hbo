# examples/sklearn_example.py - مثال عملي لتحسين نموذج Scikit-learn

import time
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification, load_breast_cancer

# تحميل البيانات
try:
    data = load_breast_cancer()
    X, y = data.data, data.target
except Exception as e:
    print(f"⚠️ لم يتم تحميل بيانات breast_cancer, سيتم استخدام بيانات مصنعة: {e}")
    X, y = make_classification(n_samples=500, n_features=15, n_informative=10, random_state=42)

def sklearn_objective(trial):
    """
    دالة الهدف لتحسين معاملات RandomForestClassifier.
    متوافقة مع كائن التجربة (TrialObject) من النظام الجديد.

    Args:
        trial (TrialObject): كائن التجربة من نظام HPO.

    Returns:
        float: متوسط الدقة (accuracy) من خلال التحقق المتقاطع.
    """
    # تعريف مساحة البحث للمعاملات باستخدام كائن التجربة
    n_estimators = trial.suggest_int('n_estimators', 10, 250)
    max_depth = trial.suggest_int('max_depth', 3, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])

    # إنشاء النموذج بالمعاملات المقترحة
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_features=max_features,
        random_state=42,
        n_jobs=-1  # استخدام كل الأنوية المتاحة
    )

    # تقييم النموذج باستخدام التحقق المتقاطع (3-fold للسرعة)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=trial.trial_id.__hash__() % (2**32 - 1))

    try:
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        accuracy = scores.mean()
    except Exception as e:
        print(f"⚠️ فشلت التجربة مع الخطأ: {e}")
        return 0.0 # إرجاع قيمة سيئة ليتم تجاهلها

    return accuracy