#!/usr/bin/env python3
"""
أمثلة متقدمة لتخصيص نظام HPO وإضافة ميزات متقدمة
Advanced Examples for HPO System Customization and Extended Features

يتطلب النظام الأساسي (hpo_system.py) ليعمل
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
import warnings
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable
import logging

# استيراد النظام الأساسي
# HPOSystem is an alias for the main Study class, used for consistency with the original script's terminology.
from hpo import Study as HPOSystem
from hpo import SearchSpace

# مكتبات اختيارية متقدمة
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.datasets import make_classification, load_digits, fetch_california_housing
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

warnings.filterwarnings('ignore')

# ======================================================================
# 1. تخصيص للمجالات المختلفة - Domain-Specific Optimizers
# ======================================================================

class FinancialModelOptimizer:
    """محسن مخصص للنماذج المالية"""

    def __init__(self, price_data, returns_data, risk_free_rate=0.02):
        """
        price_data: بيانات الأسعار
        returns_data: بيانات العوائد
        risk_free_rate: معدل العائد الخالي من المخاطر
        """
        self.price_data = np.array(price_data)
        self.returns_data = np.array(returns_data)
        self.risk_free_rate = risk_free_rate

        print(f"📈 محسن النماذج المالية")
        print(f"   بيانات الأسعار: {len(price_data)} نقطة")
        print(f"   العائد الخالي من المخاطر: {risk_free_rate:.2%}")

    def optimize_portfolio_weights(self, n_assets=5, n_trials=50):
        """تحسين أوزان المحفظة للحصول على أفضل نسبة شارب"""

        print(f"🎯 تحسين أوزان المحفظة ({n_assets} أصول)")

        def portfolio_objective(trial):
            # توليد أوزان المحفظة (يجب أن تصل لـ 1)
            weights = []
            remaining_weight = 1.0

            for i in range(n_assets - 1):
                # كل وزن يمكن أن يكون من 0 إلى الوزن المتبقي
                max_weight = min(0.5, remaining_weight)  # حد أقصى 50% لكل أصل
                weight = trial.suggest_float(f'weight_{i}', 0.0, max_weight)
                weights.append(weight)
                remaining_weight -= weight

            # الوزن الأخير هو المتبقي
            weights.append(max(0.0, remaining_weight))
            weights = np.array(weights)

            # تطبيع الأوزان لتصل لـ 1
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = np.ones(n_assets) / n_assets

            # حساب عائد ومخاطر المحفظة
            # We select columns (assets) and calculate stats along rows (time)
            asset_returns = self.returns_data[:, :n_assets]
            mean_returns = np.mean(asset_returns, axis=0)
            asset_variances = np.var(asset_returns, axis=0)

            portfolio_return = np.dot(weights, mean_returns)
            portfolio_volatility = np.sqrt(np.dot(weights**2, asset_variances))

            # نسبة شارب
            if portfolio_volatility > 0:
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            else:
                sharpe_ratio = -10  # عقوبة للتقلب الصفري

            return sharpe_ratio

        # إعداد فضاء البحث
        search_space = SearchSpace()
        for i in range(n_assets - 1):
            search_space.add_uniform(f'weight_{i}', 0.0, 0.5)

        # تشغيل التحسين
        hpo = HPOSystem(
            search_space=search_space,
            objective_function=portfolio_objective,
            direction='maximize',
            n_trials=n_trials,
            sampler='TPE',
            study_name='portfolio_optimization',
            verbose=True
        )

        best_trial = hpo.optimize()

        if best_trial:
            # استخراج الأوزان المثلى
            weights = []
            remaining_weight = 1.0

            for i in range(n_assets - 1):
                weight = best_trial.params[f'weight_{i}']
                weights.append(weight)
                remaining_weight -= weight

            weights.append(max(0.0, remaining_weight))
            weights = np.array(weights)
            weights = weights / weights.sum()  # تطبيع

            print(f"\n🏆 أفضل محفظة:")
            print(f"   نسبة شارب: {best_trial.value:.4f}")
            print(f"   أوزان المحفظة:")
            for i, weight in enumerate(weights):
                print(f"     أصل {i+1}: {weight:.3f} ({weight*100:.1f}%)")

            return weights, best_trial.value, hpo

        return None, 0, hpo

    def optimize_trading_strategy(self, n_trials=30):
        """تحسين استراتيجية التداول"""

        print("📊 تحسين استراتيجية التداول")

        def trading_objective(trial):
            # معاملات الاستراتيجية
            short_window = trial.suggest_int('short_window', 5, 20)
            long_window = trial.suggest_int('long_window', 20, 100)
            threshold = trial.suggest_float('threshold', 0.01, 0.1)
            stop_loss = trial.suggest_float('stop_loss', 0.02, 0.1)

            # تأكد من أن النافذة القصيرة أقل من الطويلة
            if short_window >= long_window:
                return -1  # عقوبة

            # محاكاة الاستراتيجية
            prices = self.price_data
            short_ma = pd.Series(prices).rolling(short_window).mean().values
            long_ma = pd.Series(prices).rolling(long_window).mean().values

            positions = np.zeros(len(prices))
            returns = np.zeros(len(prices))

            for i in range(long_window, len(prices)):
                # إشارة الشراء/البيع
                if short_ma[i] > long_ma[i] * (1 + threshold):
                    positions[i] = 1  # شراء
                elif short_ma[i] < long_ma[i] * (1 - threshold):
                    positions[i] = -1  # بيع
                else:
                    positions[i] = positions[i-1]  # الاحتفاظ بالوضع

                # حساب العائد
                if i > 0:
                    price_return = (prices[i] - prices[i-1]) / prices[i-1]
                    returns[i] = positions[i-1] * price_return

                    # تطبيق stop loss
                    if abs(returns[i]) > stop_loss:
                        positions[i] = 0  # إغلاق الوضع

            # حساب الأداء
            total_return = np.sum(returns)
            volatility = np.std(returns) if np.std(returns) > 0 else 0.01
            sharpe = total_return / volatility if volatility > 0 else 0

            return sharpe

        # إعداد فضاء البحث
        search_space = SearchSpace()
        search_space.add_int('short_window', 5, 20)
        search_space.add_int('long_window', 20, 100)
        search_space.add_uniform('threshold', 0.01, 0.1)
        search_space.add_uniform('stop_loss', 0.02, 0.1)

        # تشغيل التحسين
        hpo = HPOSystem(
            search_space=search_space,
            objective_function=trading_objective,
            direction='maximize',
            n_trials=n_trials,
            sampler='TPE',
            study_name='trading_strategy',
            verbose=True
        )

        best_trial = hpo.optimize()

        if best_trial:
            print(f"\n🎯 أفضل استراتيجية:")
            print(f"   نسبة شارب: {best_trial.value:.4f}")
            print(f"   النافذة القصيرة: {best_trial.params['short_window']} يوم")
            print(f"   النافذة الطويلة: {best_trial.params['long_window']} يوم")
            print(f"   العتبة: {best_trial.params['threshold']:.3f}")
            print(f"   وقف الخسارة: {best_trial.params['stop_loss']:.3f}")

        return best_trial, hpo

# ======================================================================
# 4. التنفيذ الرئيسي - Main Execution
# ======================================================================

def run_all_examples():
    """تشغيل جميع الأمثلة المتقدمة"""

    print("\n" + "="*70)
    print("🚀 بدء تشغيل الأمثلة المتقدمة لنظام HPO")
    print("="*70)

    # --- 1. مثال محسن النماذج المالية ---
    print("\n\n" + "-"*60)
    print("📈 1. مثال محسن النماذج المالية")
    print("-"*60)
    # بيانات مالية وهمية
    np.random.seed(42)
    price_data = np.random.randn(252).cumsum() + 100  # 1 سنة من بيانات الأسعار اليومية
    returns_data = pd.DataFrame(np.random.randn(252, 10) * 0.01, columns=[f'Asset_{i}' for i in range(10)])

    financial_optimizer = FinancialModelOptimizer(price_data, returns_data, risk_free_rate=0.03)

    # تحسين أوزان المحفظة
    financial_optimizer.optimize_portfolio_weights(n_assets=5, n_trials=30)

    # تحسين استراتيجية التداول
    financial_optimizer.optimize_trading_strategy(n_trials=25)

    # --- 2. مثال محسن نماذج معالجة اللغات الطبيعية ---
    print("\n\n" + "-"*60)
    print("🔤 2. مثال محسن نماذج NLP (محاكاة)")
    print("-"*60)
    # بيانات وهمية
    dummy_texts = ["هذا نص تجريبي رائع", "أكره هذا المنتج السيء", "هذا جيد جداً", "لا بأس به"] * 100
    dummy_labels = [1, 0, 1, 0] * 100
    nlp_optimizer = NLPModelOptimizer(dummy_texts, dummy_labels, vocab_size=5000)
    nlp_optimizer.optimize_text_preprocessing(n_trials=15)

    # --- 3. مثال محسن نماذج الرؤية الحاسوبية ---
    print("\n\n" + "-"*60)
    print("🖼️ 3. مثال محسن نماذج الرؤية الحاسوبية (محاكاة)")
    print("-"*60)
    # بيانات وهمية (في الواقع ستكون صور)
    dummy_images = [np.zeros((64, 64, 3)) for _ in range(200)]
    dummy_labels_cv = [0, 1] * 100
    cv_optimizer = ComputerVisionOptimizer(dummy_images, dummy_labels_cv, input_shape=(64, 64, 3))
    cv_optimizer.optimize_data_augmentation(n_trials=15)

    # --- 4. مثال التحسين متعدد الأهداف ---
    print("\n\n" + "-"*60)
    print("🎯 4. مثال التحسين متعدد الأهداف (محاكاة)")
    print("-"*60)
    # أوزان مختلفة للأهداف
    tradeoff_weights = {'accuracy': 0.6, 'speed': 0.25, 'memory': 0.15}
    multi_obj_optimizer = MultiObjectiveOptimizer(objectives_weights=tradeoff_weights)
    multi_obj_optimizer.optimize_model_tradeoffs(n_trials=20)

    # --- 5. مثال تحسين نماذج Scikit-learn ---
    if HAS_SKLEARN:
        print("\n\n" + "-"*60)
        print("🤖 5. مثال تحسين نماذج Scikit-learn (عملي)")
        print("-"*60)
        # استخدام بيانات Digits للتبسيط
        X, y = load_digits(return_X_y=True)

        sklearn_optimizer = SKLearnOptimizer(X, y, cv_folds=4)

        # تحسين RandomForest
        print("\n--- تحسين RandomForest ---")
        sklearn_optimizer.optimize_random_forest(n_trials=25)

        # تحسين SVC
        print("\n--- تحسين SVC ---")
        sklearn_optimizer.optimize_svc(n_trials=30)
    else:
        print("\n\n" + "-"*60)
        print("🤖 5. تخطي مثال Scikit-learn (المكتبة غير مثبتة)")
        print("-"*60)

    print("\n\n" + "="*70)
    print("🎉 اكتمل تشغيل جميع الأمثلة المتقدمة بنجاح!")
    print("="*70)

if __name__ == '__main__':
    # This script is intended to be run via `run_hpo.py`, which handles the path setup.
    # This block is for direct execution testing.
    try:
        run_all_examples()
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

# ======================================================================
# 3. مثال عملي: تحسين نماذج Scikit-learn
# Practical Example: Optimizing Scikit-learn Models
# ======================================================================

class SKLearnOptimizer:
    """محسن مخصص لنماذج Scikit-learn"""

    def __init__(self, X, y, cv_folds=5):
        if not HAS_SKLEARN:
            raise ImportError("❌ يتطلب Scikit-learn. يرجى التثبيت: pip install scikit-learn")

        self.X = X
        self.y = y
        self.cv_folds = cv_folds
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

        print(f"🤖 محسن نماذج Scikit-learn")
        print(f"   شكل البيانات: {self.X.shape}")
        print(f"   عدد الطيات (CV): {self.cv_folds}")

    def optimize_random_forest(self, n_trials=40):
        """تحسين معاملات RandomForestClassifier"""

        def rf_objective(trial):
            # تعريف معاملات البحث
            n_estimators = trial.suggest_int('n_estimators', 50, 500)
            max_depth = trial.suggest_int('max_depth', 5, 50)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
            criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

            # إنشاء النموذج
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                random_state=42,
                n_jobs=-1
            )

            # تقييم النموذج باستخدام التحقق المتقاطع
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            score = cross_val_score(model, self.X, self.y, cv=cv, scoring='accuracy').mean()

            return score

        # إعداد فضاء البحث
        search_space = SearchSpace()
        search_space.add_int('n_estimators', 50, 500)
        search_space.add_int('max_depth', 5, 50)
        search_space.add_int('min_samples_split', 2, 20)
        search_space.add_int('min_samples_leaf', 1, 10)
        search_space.add_categorical('criterion', ['gini', 'entropy'])

        # تشغيل التحسين
        hpo = HPOSystem(
            search_space=search_space,
            objective_function=rf_objective,
            direction='maximize',
            n_trials=n_trials,
            sampler='TPE',
            study_name='sklearn_rf_optimization',
            verbose=True
        )

        best_trial = hpo.optimize()

        if best_trial:
            print(f"\n🏆 أفضل نموذج RandomForest:")
            print(f"   الدقة (CV): {best_trial.value:.4f}")
            print(f"   المعاملات:")
            for param, value in best_trial.params.items():
                print(f"     {param}: {value}")

        return best_trial, hpo

    def optimize_svc(self, n_trials=50):
        """تحسين معاملات Support Vector Classifier (SVC)"""

        def svc_objective(trial):
            # تعريف معاملات البحث
            C = trial.suggest_float('C', 1e-2, 1e2, log=True)
            kernel = trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
            gamma = trial.suggest_float('gamma', 1e-4, 1e-1, log=True)

            # درجة kernel 'poly' فقط
            degree = 3
            if kernel == 'poly':
                degree = trial.suggest_int('degree', 2, 5)

            # إنشاء النموذج
            model = SVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
                degree=degree,
                probability=True,
                random_state=42
            )

            # تقييم النموذج (مهم استخدام البيانات المقاسة لـ SVC)
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            score = cross_val_score(model, self.X_scaled, self.y, cv=cv, scoring='accuracy').mean()

            return score

        # إعداد فضاء البحث
        search_space = SearchSpace()
        search_space.add_uniform('C', 1e-2, 1e2, log=True)
        search_space.add_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
        search_space.add_uniform('gamma', 1e-4, 1e-1, log=True)
        search_space.add_int('degree', 2, 5) # سيتم تجاهله إذا لم يكن kernel 'poly'

        # تشغيل التحسين
        hpo = HPOSystem(
            search_space=search_space,
            objective_function=svc_objective,
            direction='maximize',
            n_trials=n_trials,
            sampler='TPE',
            study_name='sklearn_svc_optimization',
            verbose=True
        )

        best_trial = hpo.optimize()

        if best_trial:
            print(f"\n🏆 أفضل نموذج SVC:")
            print(f"   الدقة (CV): {best_trial.value:.4f}")
            print(f"   المعاملات:")
            for param, value in best_trial.params.items():
                # لا تطبع 'degree' إذا لم يكن Kernel هو 'poly'
                if param == 'degree' and best_trial.params.get('kernel') != 'poly':
                    continue
                print(f"     {param}: {value}")

        return best_trial, hpo

class NLPModelOptimizer:
    """محسن مخصص لنماذج معالجة اللغات الطبيعية"""

    def __init__(self, text_data, labels, vocab_size=10000):
        self.text_data = text_data
        self.labels = labels
        self.vocab_size = vocab_size

        print(f"🔤 محسن نماذج NLP")
        print(f"   عدد النصوص: {len(text_data)}")
        print(f"   عدد الفئات: {len(set(labels))}")
        print(f"   حجم المفردات: {vocab_size}")

    def optimize_text_preprocessing(self, n_trials=25):
        """تحسين معاملات معالجة النصوص"""

        def preprocessing_objective(trial):
            # معاملات المعالجة
            min_word_freq = trial.suggest_int('min_word_freq', 1, 10)
            max_features = trial.suggest_int('max_features', 1000, self.vocab_size)
            ngram_range_max = trial.suggest_int('ngram_range_max', 1, 3)
            remove_stopwords = trial.suggest_categorical('remove_stopwords', [True, False])
            lowercase = trial.suggest_categorical('lowercase', [True, False])

            # محاكاة معالجة النصوص وتقييم الأداء
            # (في التطبيق الحقيقي، ستستخدم TfidfVectorizer أو مشابه)

            # محاكاة الأداء بناء على المعاملات
            base_score = 0.75

            # تأثير معاملات المعالجة
            if min_word_freq > 5:
                base_score += 0.05  # إزالة الكلمات النادرة تحسن الأداء

            if max_features > 5000:
                base_score += 0.08  # مفردات أكبر = تمثيل أفضل
            elif max_features < 2000:
                base_score -= 0.05  # مفردات صغيرة جداً

            if ngram_range_max > 1:
                base_score += 0.06  # n-grams تحسن فهم السياق

            if remove_stopwords:
                base_score += 0.03  # إزالة stop words مفيدة عادة

            if lowercase:
                base_score += 0.02  # التحويل لأحرف صغيرة يوحد النص

            # ضوضاء واقعية
            noise = np.random.normal(0, 0.05)
            final_score = base_score + noise

            # محاكاة وقت المعالجة
            processing_time = 0.1 + (max_features / 10000) * 0.3
            time.sleep(processing_time)

            return max(0.5, min(0.98, final_score))

        # إعداد فضاء البحث
        search_space = SearchSpace()
        search_space.add_int('min_word_freq', 1, 10)
        search_space.add_int('max_features', 1000, self.vocab_size)
        search_space.add_int('ngram_range_max', 1, 3)
        search_space.add_categorical('remove_stopwords', [True, False])
        search_space.add_categorical('lowercase', [True, False])

        # تشغيل التحسين
        hpo = HPOSystem(
            search_space=search_space,
            objective_function=preprocessing_objective,
            direction='maximize',
            n_trials=n_trials,
            sampler='TPE',
            study_name='nlp_preprocessing',
            verbose=True
        )

        best_trial = hpo.optimize()

        if best_trial:
            print(f"\n🎯 أفضل إعدادات معالجة النص:")
            print(f"   الدقة: {best_trial.value:.4f}")
            print(f"   تكرار الكلمة الأدنى: {best_trial.params['min_word_freq']}")
            print(f"   أقصى عدد ميزات: {best_trial.params['max_features']:,}")
            print(f"   أقصى n-gram: {best_trial.params['ngram_range_max']}")
            print(f"   إزالة stop words: {'نعم' if best_trial.params['remove_stopwords'] else 'لا'}")
            print(f"   أحرف صغيرة: {'نعم' if best_trial.params['lowercase'] else 'لا'}")

        return best_trial, hpo

class ComputerVisionOptimizer:
    """محسن مخصص لنماذج الرؤية الحاسوبية"""

    def __init__(self, image_data, labels, input_shape=(224, 224, 3)):
        self.image_data = image_data
        self.labels = labels
        self.input_shape = input_shape

        print(f"🖼️ محسن نماذج الرؤية الحاسوبية")
        print(f"   عدد الصور: {len(image_data) if hasattr(image_data, '__len__') else 'غير محدد'}")
        print(f"   شكل الدخل: {input_shape}")

    def optimize_data_augmentation(self, n_trials=20):
        """تحسين معاملات تحسين البيانات (Data Augmentation)"""

        def augmentation_objective(trial):
            # معاملات التحسين
            rotation_range = trial.suggest_int('rotation_range', 0, 45)
            zoom_range = trial.suggest_float('zoom_range', 0.0, 0.3)
            horizontal_flip = trial.suggest_categorical('horizontal_flip', [True, False])
            vertical_flip = trial.suggest_categorical('vertical_flip', [True, False])
            brightness_range = trial.suggest_float('brightness_range', 0.0, 0.3)
            contrast_range = trial.suggest_float('contrast_range', 0.0, 0.3)

            # محاكاة تأثير التحسين على الأداء
            base_accuracy = 0.78

            # تأثيرات إيجابية معقولة
            if 10 <= rotation_range <= 30:
                base_accuracy += 0.08  # دوران متوسط مفيد
            elif rotation_range > 30:
                base_accuracy += 0.03  # دوران كثير قد يضر

            if 0.1 <= zoom_range <= 0.2:
                base_accuracy += 0.06  # تكبير متوسط مفيد

            if horizontal_flip:
                base_accuracy += 0.05  # انعكاس أفقي عادة مفيد

            if vertical_flip and rotation_range > 0:
                base_accuracy += 0.02  # انعكاس رأسي أحياناً مفيد
            elif vertical_flip:
                base_accuracy -= 0.02  # قد يضر بدون دوران

            if 0.1 <= brightness_range <= 0.2:
                base_accuracy += 0.04  # تغيير إضاءة متوسط

            if 0.1 <= contrast_range <= 0.2:
                base_accuracy += 0.03  # تغيير تباين متوسط

            # عقوبة للمبالغة
            total_augmentation = rotation_range/45 + zoom_range + brightness_range + contrast_range
            if total_augmentation > 1.5:
                base_accuracy -= 0.1  # مبالغة في التحسين

            # ضوضاء
            noise = np.random.normal(0, 0.04)
            final_accuracy = base_accuracy + noise

            # محاكاة وقت التدريب (أكثر تحسين = وقت أطول)
            training_time = 0.5 + total_augmentation * 0.5
            time.sleep(training_time)

            return max(0.6, min(0.95, final_accuracy))

        # إعداد فضاء البحث
        search_space = SearchSpace()
        search_space.add_int('rotation_range', 0, 45)
        search_space.add_uniform('zoom_range', 0.0, 0.3)
        search_space.add_categorical('horizontal_flip', [True, False])
        search_space.add_categorical('vertical_flip', [True, False])
        search_space.add_uniform('brightness_range', 0.0, 0.3)
        search_space.add_uniform('contrast_range', 0.0, 0.3)

        # تشغيل التحسين
        hpo = HPOSystem(
            search_space=search_space,
            objective_function=augmentation_objective,
            direction='maximize',
            n_trials=n_trials,
            sampler='TPE',
            study_name='cv_augmentation',
            verbose=True
        )

        best_trial = hpo.optimize()

        if best_trial:
            print(f"\n🎯 أفضل إعدادات تحسين البيانات:")
            print(f"   الدقة: {best_trial.value:.4f}")
            print(f"   مدى الدوران: {best_trial.params['rotation_range']}°")
            print(f"   مدى التكبير: {best_trial.params['zoom_range']:.2f}")
            print(f"   انعكاس أفقي: {'نعم' if best_trial.params['horizontal_flip'] else 'لا'}")
            print(f"   انعكاس رأسي: {'نعم' if best_trial.params['vertical_flip'] else 'لا'}")
            print(f"   مدى الإضاءة: {best_trial.params['brightness_range']:.2f}")
            print(f"   مدى التباين: {best_trial.params['contrast_range']:.2f}")

        return best_trial, hpo

# ======================================================================
# 2. ميزات متقدمة - Advanced Features
# ======================================================================

class MultiObjectiveOptimizer:
    """التحسين متعدد الأهداف"""

    def __init__(self, objectives_weights=None):
        """
        objectives_weights: أوزان الأهداف المختلفة
        مثال: {'accuracy': 0.7, 'speed': 0.2, 'memory': 0.1}
        """
        self.objectives_weights = objectives_weights or {'primary': 1.0}

        print("🎯 محسن متعدد الأهداف")
        print("   الأهداف:")
        for objective, weight in self.objectives_weights.items():
            print(f"     {objective}: {weight:.1%}")

    def optimize_model_tradeoffs(self, n_trials=30):
        """تحسين المقايضات بين الدقة والسرعة واستهلاك الذاكرة"""

        def multi_objective_function(trial):
            # معاملات النموذج
            model_complexity = trial.suggest_int('model_complexity', 1, 10)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
            regularization = trial.suggest_float('regularization', 0.0, 0.1)

            # محاكاة الأهداف المختلفة
            # 1. الدقة (أعلى = أفضل)
            base_accuracy = 0.7 + (model_complexity / 10) * 0.2
            if 0.001 <= learning_rate <= 0.01:
                base_accuracy += 0.08
            if 0.01 <= regularization <= 0.05:
                base_accuracy += 0.05
            accuracy = min(0.98, base_accuracy + np.random.normal(0, 0.03))

            # 2. السرعة (وقت أقل = أفضل، نحوله لنقاط)
            base_time = 10 + model_complexity * 2 + (batch_size / 32) * 3
            speed_score = max(0.1, 1.0 / base_time)  # عكس الوقت

            # 3. استهلاك الذاكرة (أقل = أفضل، نحوله لنقاط)
            memory_usage = model_complexity * 100 + batch_size * 10
            memory_score = max(0.1, 1.0 / (memory_usage / 1000))

            # دمج الأهداف المتعددة
            if 'accuracy' in self.objectives_weights:
                weighted_score = accuracy * self.objectives_weights.get('accuracy', 0)
            else:
                weighted_score = accuracy * 0.6  # وزن افتراضي

            if 'speed' in self.objectives_weights:
                weighted_score += speed_score * self.objectives_weights.get('speed', 0)
            else:
                weighted_score += speed_score * 0.3

            if 'memory' in self.objectives_weights:
                weighted_score += memory_score * self.objectives_weights.get('memory', 0)
            else:
                weighted_score += memory_score * 0.1

            # حفظ القيم المنفصلة للتحليل
            trial.user_attrs = {
                'accuracy': accuracy,
                'speed_score': speed_score,
                'memory_score': memory_score,
                'estimated_time': base_time,
                'estimated_memory': memory_usage
            }

            return weighted_score

        # إعداد فضاء البحث
        search_space = SearchSpace()
        search_space.add_int('model_complexity', 1, 10)
        search_space.add_categorical('batch_size', [16, 32, 64, 128, 256])
        search_space.add_uniform('learning_rate', 1e-5, 1e-1, log=True)
        search_space.add_uniform('regularization', 0.0, 0.1)

        # تشغيل التحسين
        hpo = HPOSystem(
            search_space=search_space,
            objective_function=multi_objective_function,
            direction='maximize',
            n_trials=n_trials,
            sampler='TPE',
            study_name='multi_objective_tradeoffs',
            verbose=True
        )

        best_trial_result = hpo.optimize()

        # For detailed printing, access the full trial object from the study, which contains user_attrs
        best_trial_for_printing = hpo.best_trial

        if best_trial_for_printing:
            print(f"\n🏆 أفضل نموذج من حيث المقايضة:")
            print(f"   النتيجة الموزونة: {best_trial_for_printing.value:.4f}")
            print(f"   المعاملات:")
            print(f"     تعقيد النموذج: {best_trial_for_printing.params['model_complexity']}")
            print(f"     حجم الدفعة: {best_trial_for_printing.params['batch_size']}")
            print(f"     معدل التعلم: {best_trial_for_printing.params['learning_rate']:.5f}")
            print(f"     التنظيم: {best_trial_for_printing.params['regularization']:.4f}")

            print(f"\n   الأهداف المقدرة:")
            if hasattr(best_trial_for_printing, 'user_attrs') and best_trial_for_printing.user_attrs:
                print(f"     الدقة: {best_trial_for_printing.user_attrs.get('accuracy', 0):.4f}")
                print(f"     نقاط السرعة: {best_trial_for_printing.user_attrs.get('speed_score', 0):.4f} (وقت: {best_trial_for_printing.user_attrs.get('estimated_time', 0):.1f}s)")
                print(f"     نقاط الذاكرة: {best_trial_for_printing.user_attrs.get('memory_score', 0):.4f} (ذاكرة: {best_trial_for_printing.user_attrs.get('estimated_memory', 0):.1f} MB)")

        return best_trial_result, hpo