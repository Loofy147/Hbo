import subprocess
import sys
import os

# اسم ملف المتطلبات
REQUIREMENTS_FILE = "requirements.txt"

def install_requirements():
    """
    تثبيت الحزم من ملف requirements.txt.
    """
    if not os.path.exists(REQUIREMENTS_FILE):
        print(f"🛑 خطأ: ملف '{REQUIREMENTS_FILE}' غير موجود.")
        print("يرجى التأكد من وجود الملف في نفس المجلد.")
        sys.exit(1)

    print(f"🔧 جارٍ تثبيت المتطلبات من '{REQUIREMENTS_FILE}'...")

    try:
        # استخدام pip لتثبيت الحزم
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE])
        print("\n✅ تم تثبيت جميع المتطلبات بنجاح!")
    except subprocess.CalledProcessError as e:
        print(f"\n🛑 حدث خطأ أثناء تثبيت المتطلبات: {e}")
        print("يرجى المحاولة مرة أخرى أو تثبيت المتطلبات يدويًا باستخدام:")
        print(f"pip install -r {REQUIREMENTS_FILE}")
        sys.exit(1)
    except FileNotFoundError:
        print("\n🛑 خطأ: لم يتم العثور على 'pip'.")
        print("يرجى التأكد من أن Python و pip مثبتان بشكل صحيح ومضافان إلى متغيرات البيئة (PATH).")
        sys.exit(1)

if __name__ == "__main__":
    print("=============================================")
    print("      🚀 مثبت المتطلبات لنظام HPO")
    print("=============================================")

    install_requirements()

    print("\n🎉 كل شيء جاهز الآن!")
    print("يمكنك الآن تشغيل النظام باستخدام 'run_hpo.py'.")