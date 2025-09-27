#!/bin/bash
# start.sh - تشغيل تفاعلي لنظام HPO

# دالة لطباعة الترويسة
print_header() {
    echo "============================================="
    echo "    🚀 نظام HPO - لوحة التحكم التفاعلية     "
    echo "============================================="
    echo " مرحباً بك! اختر إحدى الخيارات للبدء."
    echo
}

# دالة لطباعة الخيارات
print_menu() {
    echo "  1. 🔬 فحص النظام (check)"
    echo "  2. ✨ تشغيل مثال سريع (quick)"
    echo "  3. 🎯 بدء عملية تحسين كاملة (full run)"
    echo "  4. 🔧 تثبيت المتطلبات (install)"
    echo "  5.  exit"
    echo
}

# التأكد من وجود الملف الرئيسي للتشغيل
if [ ! -f "run_hpo.py" ]; then
    echo "🛑 خطأ: الملف 'run_hpo.py' غير موجود."
    echo "يرجى التأكد من أنك في المجلد الصحيح للمشروع."
    exit 1
fi

# حلقة البرنامج الرئيسية
while true; do
    print_header
    print_menu

    read -p "الرجاء إدخال اختيارك (1-5): " choice

    case $choice in
        1)
            echo "🔬 جارٍ فحص النظام..."
            python run_hpo.py --check
            ;;
        2)
            echo "✨ جارٍ تشغيل المثال السريع..."
            python run_hpo.py --quick
            ;;
        3)
            echo "🎯 جارٍ بدء عملية تحسين كاملة..."
            python run_hpo.py
            ;;
        4)
            echo "🔧 جارٍ تثبيت المتطلبات..."
            if [ -f "install.py" ]; then
                python install.py
            else
                echo "🛑 خطأ: الملف 'install.py' غير موجود."
            fi
            ;;
        5)
            echo "👋 إلى اللقاء!"
            break
            ;;
        *)
            echo "⚠️ اختيار غير صالح. الرجاء إدخال رقم بين 1 و 5."
            ;;
    esac

    echo
    read -p "اضغط على Enter للعودة إلى القائمة الرئيسية..."
    clear # لمسح الشاشة قبل عرض القائمة مرة أخرى
done