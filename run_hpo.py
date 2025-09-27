import argparse
import sys
import time

# --- استيراد المكونات الأساسية من النظام الجديد ---
try:
    # استيراد الفئات الرئيسية من hpo_production_system.py
    from hpo_production_system import HPOSystem, SearchSpace, TrialObject
    # استيراد الأمثلة الجاهزة من نفس الملف
    from hpo_production_system import example_mathematical, example_ml_simulation, example_sklearn_real
    SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"🛑 خطأ: لا يمكن استيراد المكونات الأساسية من 'hpo_production_system.py': {e}")
    print("يرجى التأكد من أن الملف موجود وأن البيئة مهيأة بشكل صحيح.")
    HPOSystem, SearchSpace = None, None
    SYSTEM_AVAILABLE = False

# --- استيراد دوال الهدف المنفصلة ---
try:
    from quick_start import mathematical_objective
    from examples.sklearn_example import sklearn_objective
    EXAMPLES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ تحذير: لم يتم العثور على بعض دوال الهدف المنفصلة: {e}")
    mathematical_objective = None
    sklearn_objective = None
    EXAMPLES_AVAILABLE = False


def run_check():
    """
    فحص سريع للنظام والمتطلبات الأساسية.
    """
    print("=============================================")
    print("      🔬 فحص سريع لنظام HPO (النسخة الجديدة)")
    print("=============================================")
    all_ok = True

    # 1. التحقق من وجود الملفات الأساسية
    print("\n[1/3] 📂 التحقق من وجود الملفات الأساسية...")
    required_files = ["run_hpo.py", "hpo_production_system.py", "install.py", "requirements.txt"]
    for f in required_files:
        try:
            with open(f, 'r'): pass
            print(f"  ✅ {f}")
        except FileNotFoundError:
            print(f"  🛑 {f} (غير موجود!)")
            all_ok = False

    # 2. التحقق من استيراد المكتبات الجديدة
    print("\n[2/3] 📦 التحقق من استيراد المكتبات الأساسية...")
    libs_to_check = {'numpy': 'numpy', 'pandas': 'pandas', 'sklearn': 'scikit-learn', 'matplotlib': 'matplotlib'}
    for lib, name in libs_to_check.items():
        try:
            __import__(lib)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  🛑 {name} (غير مثبت!)")
            all_ok = False

    # 3. التحقق من جاهزية النظام
    print("\n[3/3] ⚙️ التحقق من جاهزية النظام...")
    if SYSTEM_AVAILABLE and HPOSystem and SearchSpace:
        print("  ✅ تم استيراد HPOSystem و SearchSpace بنجاح.")
    else:
        print("  🛑 لم يتم استيراد HPOSystem أو SearchSpace.")
        all_ok = False

    # الخلاصة
    print("\n---------------------------------------------")
    if all_ok:
        print("🎉 فحص النظام مكتمل! كل شيء يبدو جاهزًا للانطلاق.")
    else:
        print("⚠️ تم العثور على مشاكل. يرجى مراجعة الرسائل أعلاه.")
        print("قد تحتاج إلى تشغيل 'python install.py' أو 'pip install -r requirements.txt'.")
    print("=============================================")


def run_quick_example():
    """
    تشغيل مثال رياضي بسيط وسريع باستخدام دالة الهدف المنفصلة.
    """
    if not (SYSTEM_AVAILABLE and EXAMPLES_AVAILABLE and mathematical_objective):
        print("🛑 خطأ: المثال السريع غير متوفر. تأكد من وجود كل الملفات المطلوبة.")
        return

    print("=============================================")
    print("    ✨ المثال السريع: دالة رياضية بسيطة")
    print("=============================================")

    # 1. إعداد فضاء البحث
    search_space = SearchSpace()
    search_space.add_uniform('x', -10, 10)
    search_space.add_uniform('y', -10, 10)

    # 2. تهيئة وتشغيل النظام
    hpo = HPOSystem(
        search_space=search_space,
        objective_function=mathematical_objective,
        direction='maximize',
        n_trials=30,
        n_startup_trials=5,
        sampler='TPE',
        study_name='quick_start_run'
    )

    hpo.optimize()
    # الملخص والتحليل سيتم طباعتهما من داخل hpo.print_summary()


def run_full_optimization():
    """
    تشغيل مثال تحسين كامل باستخدام Scikit-learn ودالة الهدف المنفصلة.
    """
    if not (SYSTEM_AVAILABLE and EXAMPLES_AVAILABLE and sklearn_objective):
        print("🛑 خطأ: مثال Scikit-learn غير متوفر. تأكد من وجود كل الملفات المطلوبة.")
        return

    print("=======================================================")
    print("  🎯 التحسين الكامل: نموذج RandomForest على بيانات Iris")
    print("=======================================================")

    # 1. إعداد فضاء البحث
    search_space = SearchSpace()
    search_space.add_int('n_estimators', 10, 250)
    search_space.add_int('max_depth', 3, 30)
    search_space.add_int('min_samples_split', 2, 20)
    search_space.add_categorical('max_features', ['sqrt', 'log2'])

    # 2. تهيئة وتشغيل النظام
    hpo = HPOSystem(
        search_space=search_space,
        objective_function=sklearn_objective,
        direction='maximize',
        n_trials=25,
        n_startup_trials=5,
        sampler='TPE',
        study_name='full_sklearn_run'
    )

    hpo.optimize()
    # الملخص والتحليل سيتم طباعتهما من داخل hpo.print_summary()

def run_embedded_examples():
    """
    تشغيل الأمثلة المدمجة مباشرة من hpo_production_system.py
    """
    if not SYSTEM_AVAILABLE:
        print("🛑 خطأ: النظام الأساسي غير متوفر.")
        return

    print("\n\n--- [ المثال الأول المدمج: رياضي بسيط ] ---\n")
    hpo_math = example_mathematical()
    if hpo_math and hpo_math.stats['n_complete'] > 0:
        hpo_math.plot_optimization_history()

    print("\n\n--- [ المثال الثاني المدمج: Scikit-learn حقيقي ] ---\n")
    hpo_sklearn = example_sklearn_real()
    if hpo_sklearn and hpo_sklearn.stats['n_complete'] > 0:
        hpo_sklearn.plot_optimization_history()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🚀 نظام HPO - أداة تحسين المعاملات الفائقة (النسخة الجديدة)",
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
        help='✨ تشغيل مثال رياضي سريع (باستخدام quick_start.py).'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='🎯 تشغيل مثال scikit-learn كامل (باستخدام examples/sklearn_example.py).'
    )
    parser.add_argument(
        '--embedded',
        action='store_true',
        help='🌟 تشغيل الأمثلة المدمجة في hpo_production_system.py.'
    )

    args = parser.parse_args()

    if args.check:
        run_check()
    elif args.quick:
        run_quick_example()
    elif args.full:
        run_full_optimization()
    elif args.embedded:
        run_embedded_examples()
    else:
        # إذا لم يتم تمرير أي وسيط، نعرض المساعدة
        parser.print_help()