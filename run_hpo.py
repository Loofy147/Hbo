import argparse
import sys
import time

# محاولة استيراد النظام الأساسي
try:
    from hpo_production_system import HPOSystem
except ImportError:
    print("🛑 خطأ: لا يمكن استيراد 'HPOSystem' من 'hpo_production_system.py'.")
    print("يرجى التأكد من أن الملف موجود وأن البيئة مهيأة بشكل صحيح.")
    # سنسمح للبرنامج بالاستمرار في حالة --check أو --quick لتقديم المساعدة
    HPOSystem = None

# استيراد الأمثلة
try:
    from quick_start import mathematical_example
    from examples.sklearn_example import sklearn_objective
except ImportError as e:
    print(f"⚠️ تحذير: لم يتم العثور على بعض الأمثلة: {e}")
    mathematical_example = None
    sklearn_objective = None


def run_check():
    """
    فحص سريع للنظام والمتطلبات الأساسية.
    """
    print("=============================================")
    print("      🔬 فحص سريع لنظام HPO")
    print("=============================================")
    all_ok = True

    # 1. التحقق من وجود الملفات الأساسية
    print("\n[1/3] 📂 التحقق من وجود الملفات الأساسية...")
    required_files = [
        "run_hpo.py", "hpo_production_system.py", "install.py",
        "requirements.txt", "quick_start.py", "examples/sklearn_example.py"
    ]
    for f in required_files:
        try:
            # هذا مجرد فحص بسيط لوجود الملف
            with open(f, 'r'): pass
            print(f"  ✅ {f}")
        except FileNotFoundError:
            print(f"  🛑 {f} (غير موجود!)")
            all_ok = False

    # 2. التحقق من استيراد المكتبات
    print("\n[2/3] 📦 التحقق من استيراد المكتبات الأساسية...")
    try:
        import optuna
        print("  ✅ Optuna")
    except ImportError:
        print("  🛑 Optuna (غير مثبت!)")
        all_ok = False

    try:
        import sklearn
        print("  ✅ Scikit-learn")
    except ImportError:
        print("  🛑 Scikit-learn (غير مثبت!)")
        all_ok = False

    try:
        import numpy
        print("  ✅ NumPy")
    except ImportError:
        print("  🛑 NumPy (غير مثبت!)")
        all_ok = False

    # 3. التحقق من جاهزية النظام
    print("\n[3/3] ⚙️ التحقق من جاهزية النظام...")
    if HPOSystem:
        print("  ✅ تم استيراد HPOSystem بنجاح.")
    else:
        print("  🛑 لم يتم استيراد HPOSystem.")
        all_ok = False

    # الخلاصة
    print("\n---------------------------------------------")
    if all_ok:
        print("🎉 فحص النظام مكتمل! كل شيء يبدو جاهزًا للانطلاق.")
    else:
        print("⚠️ تم العثور على مشاكل. يرجى مراجعة الرسائل أعلاه.")
        print("قد تحتاج إلى تشغيل 'python install.py' لتثبيت المتطلبات.")
    print("=============================================")


def run_quick_example():
    """
    تشغيل مثال رياضي بسيط وسريع.
    """
    if not mathematical_example or not HPOSystem:
        print("🛑 خطأ: المثال السريع غير متوفر. تأكد من وجود 'quick_start.py' و 'hpo_production_system.py'.")
        return

    print("=============================================")
    print("    ✨ المثال السريع: دالة رياضية بسيطة")
    print("=============================================")
    print("الهدف: إيجاد x و y الذين يعظمان الدالة f(x, y) = -(x-3)^2 - (y+2)^2 + 10")
    print("الحل الأمثل المعروف: x=3, y=-2, القيمة القصوى=10")
    print("---------------------------------------------")

    # تعريف مساحة البحث
    search_space = {
        "x": ("float", -5, 5),
        "y": ("float", -5, 5)
    }

    # تهيئة النظام
    hpo = HPOSystem(
        search_space=search_space,
        objective_function=mathematical_example,
        n_trials=50,
        direction="maximize"
    )

    print("🚀 بدء عملية التحسين...")
    start_time = time.time()
    best_params, best_value = hpo.optimize()
    duration = time.time() - start_time

    print("\n🎉 انتهت عملية التحسين!")
    print(f"⏱️ المدة: {duration:.2f} ثانية")
    print("---------------------------------------------")
    print("📊 النتائج:")
    print(f"  🎯 أفضل قيمة تم العثور عليها: {best_value:.4f}")
    print(f"  ⚙️ أفضل المعاملات:")
    print(f"    - x: {best_params['x']:.4f} (المتوقع: 3.0)")
    print(f"    - y: {best_params['y']:.4f} (المتوقع: -2.0)")

    # التحقق من الدقة
    if abs(best_params['x'] - 3) < 0.1 and abs(best_params['y'] - (-2)) < 0.1:
        print("\n✅ تم العثور على الحل الأمثل بدقة عالية!")
    else:
        print("\n⚠️ الحل قريب من الأمثل. زيادة عدد التجارب (n_trials) قد تحسن الدقة.")


def run_full_optimization():
    """
    تشغيل مثال تحسين كامل باستخدام Scikit-learn.
    """
    if not sklearn_objective or not HPOSystem:
        print("🛑 خطأ: مثال Scikit-learn غير متوفر. تأكد من وجود 'examples/sklearn_example.py'.")
        return

    print("=======================================================")
    print("  🎯 التحسين الكامل: نموذج RandomForest على بيانات Iris")
    print("=======================================================")
    print("الهدف: إيجاد أفضل المعاملات لنموذج RandomForestClassifier لتحقيق أعلى دقة.")
    print("-------------------------------------------------------")

    # تعريف مساحة البحث
    search_space = {
        "n_estimators": ("int", 10, 300),
        "max_depth": ("int", 3, 20),
        "min_samples_split": ("int", 2, 10),
        "max_features": ("categorical", ["sqrt", "log2"])
    }

    # تهيئة النظام مع ميزات متقدمة
    hpo = HPOSystem(
        search_space=search_space,
        objective_function=sklearn_objective,
        n_trials=100,
        direction="maximize",
        sampler="TPE",  # استخدام TPE sampler
        pruner="ASHA",   # استخدام ASHA pruner
        n_jobs=2,        # استخدام معالجين بالتوازي
    )

    print("🚀 بدء عملية التحسين (قد تستغرق بضع دقائق)...")
    start_time = time.time()
    best_params, best_value = hpo.optimize()
    duration = time.time() - start_time

    print("\n🎉 انتهت عملية التحسين!")
    print(f"⏱️ المدة: {duration:.2f} ثانية")
    print("---------------------------------------------")
    print("📊 النتائج:")
    print(f"  🎯 أفضل دقة تم تحقيقها: {best_value:.4f}")
    print(f"  ⚙️ أفضل المعاملات:")
    for param, value in best_params.items():
        print(f"    - {param}: {value}")

    print("\n📈 يمكنك الآن مراجعة التقارير والرسوم البيانية التي تم إنشاؤها.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🚀 نظام HPO - أداة تحسين المعاملات الفائقة",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--check',
        action='store_true',
        help='🔬 فحص سريع للنظام والمتطلبات.'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='✨ تشغيل مثال رياضي سريع للتحقق من أن كل شيء يعمل.'
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        # لا توجد وسائط، تشغيل التحسين الكامل
        run_full_optimization()
    elif args.check:
        run_check()
    elif args.quick:
        run_quick_example()
    else:
        parser.print_help()