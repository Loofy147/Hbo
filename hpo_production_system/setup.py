from setuptools import setup, find_packages
import os

# قراءة المتطلبات من ملف requirements.txt
def read_requirements():
    """
    قراءة ملف المتطلبات.
    """
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if not os.path.exists(requirements_path):
        return []
    with open(requirements_path, 'r') as f:
        # إزالة التعليقات والأسطر الفارغة
        return [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith('#')
        ]

setup(
    name="hpo_production_system",
    version="1.0.0",
    author="Jules",
    author_email="jules@example.com",
    description="نظام متكامل لتحسين المعاملات الفائقة (HPO) جاهز للإنتاج",
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/example/hpo_system",  # استبدل بالرابط الفعلي
    packages=find_packages(exclude=["tests", "docs"]),
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            'hpo-run=run_hpo:main',  # مثال على كيفية إنشاء أمر في الـ command line
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
    include_package_data=True, # ليشمل ملفات غير برمجية مثل .yaml
    package_data={
        '': ['*.yaml', '*.md'],
    },
)