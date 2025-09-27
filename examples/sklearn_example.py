# examples/sklearn_example.py - مثال عملي لتحسين نموذج Scikit-learn

import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import numpy as np

# تحميل بيانات Iris كمثال
X, y = sklearn.datasets.load_iris(return_X_y=True)

def sklearn_objective(trial):
    """
    دالة الهدف لتحسين معاملات RandomForestClassifier.

    Args:
        trial (optuna.trial.Trial): كائن التجربة من Optuna.

    Returns:
        float: متوسط الدقة (accuracy) من خلال التحقق المتقاطع (cross-validation).
    """
    # تعريف مساحة البحث للمعاملات
    # لاحظ أن مساحة البحث هنا معرفة داخل الدالة،
    # بينما في run_hpo.py يتم تمريرها من الخارج. كلاهما ممكن.
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'random_state': 42  # للتأكد من أن النتائج قابلة للتكرار
    }

    # إنشاء النموذج بالمعاملات المقترحة
    model = sklearn.ensemble.RandomForestClassifier(**params)

    # تقييم النموذج باستخدام التحقق المتقاطع
    # نستخدم 3-fold cross-validation
    # ASHA Pruner يحتاج إلى تقييمات وسيطة، ولكن للتبسيط هنا سنعيد فقط المتوسط النهائي.
    # في التطبيقات المتقدمة، يمكن تعديل هذا الجزء للإبلاغ عن النتائج بعد كل fold.
    try:
        scores = sklearn.model_selection.cross_val_score(model, X, y, n_jobs=-1, cv=3)
        accuracy = scores.mean()
    except Exception as e:
        # في حالة فشل تدريب النموذج (مثلاً، معاملات غير متوافقة)
        print(f"⚠️ فشلت التجربة بالمعاملات {params} مع الخطأ: {e}")
        # إرجاع قيمة سيئة جدًا ليتم تجاهلها
        return 0.0

    # ASHA Pruning (اختياري لكن مهم)
    # للإبلاغ عن النتائج الوسيطة للـ pruner
    trial.report(accuracy, step=1)

    # التعامل مع طلبات الـ Pruning
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return accuracy