 feature/hpo-production-system
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
=======
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sys
import os

# This is a temporary measure to make the HPOSystem available for the example.
# In a real package, this would be handled by the setup.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Assuming hpo_core.py is in the parent directory.
# If you have a proper installation, you would use:
# from hpo_production_system import HPOSystem, SearchSpace
from hpo_core import HPOSystem, SearchSpace


def optimize_random_forest():
    """تحسين Random Forest Classifier"""

    print("\n🎯 تحسين Random Forest Classifier")
    print("=" * 35)

    # تحميل البيانات
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        n_clusters_per_class=2,
        random_state=42
    )

    print(f"البيانات: {X.shape[0]} عينة، {X.shape[1]} ميزة")

    def objective(trial):
        # معاملات للتحسين
        n_estimators = trial['suggest_int']('n_estimators', 10, 300)
        max_depth = trial['suggest_int']('max_depth', 3, 20)
        min_samples_split = trial['suggest_int']('min_samples_split', 2, 20)
        min_samples_leaf = trial['suggest_int']('min_samples_leaf', 1, 20)
        max_features = trial['suggest_categorical']('max_features', ['sqrt', 'log2', None, 0.5, 0.8])
        bootstrap = trial['suggest_categorical']('bootstrap', [True, False])

        # إنشاء النموذج
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=42,
            n_jobs=-1
        )

        # التقييم
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

        return scores.mean()

    # إعداد فضاء البحث
    search_space = SearchSpace()
    search_space.add_int('n_estimators', 10, 300)
    search_space.add_int('max_depth', 3, 20)
    search_space.add_int('min_samples_split', 2, 20)
    search_space.add_int('min_samples_leaf', 1, 20)
    search_space.add_categorical('max_features', ['sqrt', 'log2', None, 0.5, 0.8])
    search_space.add_categorical('bootstrap', [True, False])

    # تشغيل التحسين
    hpo = HPOSystem(
        search_space=search_space,
        objective_function=objective,
        direction='maximize',
        n_trials=50,
        n_startup_trials=10,
        sampler='TPE',
        pruner='MedianPruner',
        study_name='random_forest_optimization',
        verbose=True
    )

    best_trial = hpo.optimize()

    # النتائج
    print(f"\n🎯 أفضل دقة: {best_trial.value:.4f}")
    print("أفضل معاملات:")
    for name, value in best_trial.params.items():
        print(f"  {name}: {value}")

    return hpo, best_trial

def optimize_svm():
    """تحسين Support Vector Machine"""

    print("\n🎯 تحسين Support Vector Machine")
    print("=" * 35)

    # تحميل البيانات (أصغر للـ SVM)
    data = load_digits()
    X, y = data.data, data.target

    # تقليل حجم البيانات للسرعة
    from sklearn.model_selection import train_test_split
    X_sample, _, y_sample, _ = train_test_split(
        X, y, train_size=0.3, random_state=42, stratify=y
    )

    print(f"البيانات: {X_sample.shape[0]} عينة، {X_sample.shape[1]} ميزة")

    def objective(trial):
        # معاملات للتحسين
        C = trial['suggest_float']('C', 1e-3, 1e3, log=True)
        gamma = trial['suggest_categorical']('gamma', ['scale', 'auto']) \
               if trial['suggest_categorical']('gamma_type', ['preset', 'custom']) == 'preset' \
               else trial['suggest_float']('gamma_value', 1e-5, 1e2, log=True)
        kernel = trial['suggest_categorical']('kernel', ['rbf', 'poly', 'sigmoid'])

        if kernel == 'poly':
            degree = trial['suggest_int']('degree', 2, 5)
        else:
            degree = 3

        # إنشاء pipeline مع تقييس
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(C=C, gamma=gamma, kernel=kernel, degree=degree, random_state=42))
        ])

        # التقييم (3-fold للسرعة)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_sample, y_sample, cv=cv, scoring='accuracy')

        return scores.mean()

    # إعداد فضاء البحث
    search_space = SearchSpace()
    search_space.add_uniform('C', 1e-3, 1e3, log=True)
    search_space.add_categorical('gamma_type', ['preset', 'custom'])
    search_space.add_categorical('gamma', ['scale', 'auto'])
    search_space.add_uniform('gamma_value', 1e-5, 1e2, log=True)
    search_space.add_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
    search_space.add_int('degree', 2, 5)

    # تشغيل التحسين
    hpo = HPOSystem(
        search_space=search_space,
        objective_function=objective,
        direction='maximize',
        n_trials=30,
        n_startup_trials=5,
        sampler='TPE',
        study_name='svm_optimization',
        verbose=True
    )

    best_trial = hpo.optimize()

    # النتائج
    print(f"\n🎯 أفضل دقة: {best_trial.value:.4f}")
    print("أفضل معاملات:")
    for name, value in best_trial.params.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.6f}")
        else:
            print(f"  {name}: {value}")

    return hpo, best_trial

def compare_models():
    """مقارنة عدة نماذج محسنة"""

    print("\n🔬 مقارنة النماذج المحسنة")
    print("=" * 30)

    # تحميل بيانات موحدة للمقارنة
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )

    print(f"بيانات المقارنة: {X.shape[0]} عينة، {X.shape[1]} ميزة")

    results = {}

    # تحسين Random Forest
    print("\n1️⃣ تحسين Random Forest...")
    def rf_objective(trial):
        model = RandomForestClassifier(
            n_estimators=trial['suggest_int']('n_estimators', 10, 200),
            max_depth=trial['suggest_int']('max_depth', 3, 15),
            min_samples_split=trial['suggest_int']('min_samples_split', 2, 15),
            random_state=42
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        return cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()

    rf_space = SearchSpace()
    rf_space.add_int('n_estimators', 10, 200)
    rf_space.add_int('max_depth', 3, 15)
    rf_space.add_int('min_samples_split', 2, 15)

    rf_hpo = HPOSystem(
        search_space=rf_space,
        objective_function=rf_objective,
        direction='maximize',
        n_trials=20,
        study_name='rf_comparison',
        verbose=False
    )

    rf_best = rf_hpo.optimize()
    results['Random Forest'] = rf_best.value

    # تحسين SVM مبسط
    print("2️⃣ تحسين SVM...")
    def svm_objective(trial):
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                C=trial['suggest_float']('C', 1e-2, 1e2, log=True),
                gamma=trial['suggest_float']('gamma', 1e-4, 1e1, log=True),
                random_state=42
            ))
        ])
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        return cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()

    svm_space = SearchSpace()
    svm_space.add_uniform('C', 1e-2, 1e2, log=True)
    svm_space.add_uniform('gamma', 1e-4, 1e1, log=True)

    svm_hpo = HPOSystem(
        search_space=svm_space,
        objective_function=svm_objective,
        direction='maximize',
        n_trials=20,
        study_name='svm_comparison',
        verbose=False
    )

    svm_best = svm_hpo.optimize()
    results['SVM'] = svm_best.value

    # طباعة النتائج
    print("\n📊 نتائج المقارنة:")
    for model_name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model_name}: {accuracy:.4f}")

    best_model = max(results.items(), key=lambda x: x[1])
    print(f"\n🏆 أفضل نموذج: {best_model[0]} بدقة {best_model[1]:.4f}")

    return results

def main():
    """الدالة الرئيسية"""

    print("🔬 أمثلة متقدمة لتحسين نماذج Scikit-learn")
    print("=" * 50)

    try:
        print("\nاختر المثال:")
        print("1. تحسين Random Forest")
        print("2. تحسين SVM")
        print("3. مقارنة النماذج")
        print("4. تشغيل جميع الأمثلة")

        choice = input("\nالاختيار (1/2/3/4): ").strip()

        if choice == '1':
            hpo, best = optimize_random_forest()
        elif choice == '2':
            hpo, best = optimize_svm()
        elif choice == '3':
            compare_models()
        elif choice == '4':
            optimize_random_forest()
            optimize_svm()
            compare_models()
        else:
            print("تم اختيار المثال الأول افتراضياً...")
            hpo, best = optimize_random_forest()

        print("\n✅ انتهت الأمثلة بنجاح!")

    except KeyboardInterrupt:
        print("\n⏹️ تم إيقاف البرنامج")
    except Exception as e:
        print(f"❌ خطأ: {e}")

if __name__ == '__main__':
    mai>> main
