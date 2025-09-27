#!/usr/bin/env python3
"""
نظام ضبط المعاملات الفائقة المتقدم - نسخة مكتملة وجاهزة للتشغيل
Advanced Hyperparameter Optimization System - Complete Working Version

هذا ملف واحد مكتمل يحتوي على جميع المكونات
يمكن تشغيله مباشرة بدون ملفات إضافية
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import time
import json
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# للتوافق مع scikit-learn
try:
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification, load_breast_cancer
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ==============================================================================
# 1. فئات البيانات الأساسية
# ==============================================================================

class TrialResult:
    """نتيجة تجربة واحدة"""
    def __init__(self, trial_id, params, value, state='COMPLETE', duration=0.0):
        self.trial_id = trial_id
        self.params = params.copy()
        self.value = float(value)
        self.state = state
        self.duration = duration
        self.timestamp = time.time()

class SearchSpace:
    """فضاء البحث للمعاملات"""
    def __init__(self):
        self.params = {}
        self.param_types = {}

    def add_uniform(self, name, low, high, log=False):
        """إضافة معامل مستمر"""
        self.params[name] = {'type': 'uniform', 'low': low, 'high': high, 'log': log}
        self.param_types[name] = 'uniform'
        return self

    def add_int(self, name, low, high):
        """إضافة معامل صحيح"""
        self.params[name] = {'type': 'int', 'low': low, 'high': high}
        self.param_types[name] = 'int'
        return self

    def add_categorical(self, name, choices):
        """إضافة معامل فئوي"""
        self.params[name] = {'type': 'categorical', 'choices': choices}
        self.param_types[name] = 'categorical'
        return self

    def sample(self, n_samples=1):
        """أخذ عينات عشوائية"""
        samples = []
        for _ in range(n_samples):
            sample = {}
            for name, config in self.params.items():
                if config['type'] == 'uniform':
                    if config.get('log', False):
                        sample[name] = np.exp(np.random.uniform(
                            np.log(config['low']), np.log(config['high'])))
                    else:
                        sample[name] = np.random.uniform(config['low'], config['high'])
                elif config['type'] == 'int':
                    sample[name] = np.random.randint(config['low'], config['high'] + 1)
                elif config['type'] == 'categorical':
                    sample[name] = np.random.choice(config['choices'])
            samples.append(sample)
        return samples[0] if n_samples == 1 else samples

# ==============================================================================
# 2. نماذج العينات (Samplers)
# ==============================================================================

class TPESampler:
    """Tree-structured Parzen Estimator Sampler"""
    def __init__(self, n_startup_trials=10, n_ei_candidates=24, gamma=0.25):
        self.n_startup_trials = n_startup_trials
        self.n_ei_candidates = n_ei_candidates
        self.gamma = gamma

    def suggest(self, trials, search_space):
        """اقتراح معاملات جديدة"""
        complete_trials = [t for t in trials if t.state == 'COMPLETE']

        if len(complete_trials) < self.n_startup_trials:
            return search_space.sample()

        # ترتيب التجارب حسب الأداء
        sorted_trials = sorted(complete_trials, key=lambda t: t.value, reverse=True)

        # تقسيم إلى جيدة وسيئة
        n_good = max(1, int(len(sorted_trials) * self.gamma))
        good_trials = sorted_trials[:n_good]
        bad_trials = sorted_trials[n_good:]

        # توليد مرشحين وتقييمهم
        best_params = None
        best_ei = -np.inf

        for _ in range(self.n_ei_candidates):
            candidate = search_space.sample()

            # حساب كثافة الاحتمال للمجموعات الجيدة والسيئة
            good_density = self._compute_density(candidate, good_trials, search_space)
            bad_density = self._compute_density(candidate, bad_trials, search_space)

            if bad_density > 0:
                ei = good_density / bad_density
                if ei > best_ei:
                    best_ei = ei
                    best_params = candidate

        return best_params or search_space.sample()

    def _compute_density(self, candidate, trials, search_space):
        """حساب كثافة الاحتمال"""
        if not trials:
            return 1.0

        log_density = 0.0

        for name, value in candidate.items():
            param_config = search_space.params[name]
            param_values = [t.params[name] for t in trials if name in t.params]

            if not param_values:
                continue

            if param_config['type'] in ['uniform', 'int']:
                # استخدام Kernel Density Estimation مبسط
                if len(param_values) > 1:
                    # حساب المسافة من القيم الموجودة
                    distances = [abs(value - pv) for pv in param_values]
                    min_distance = min(distances) + 1e-10
                    density = 1.0 / min_distance
                    log_density += np.log(max(density, 1e-10))
                else:
                    log_density += 0.0

            elif param_config['type'] == 'categorical':
                # حساب التكرار النسبي
                count = param_values.count(value)
                prob = (count + 1) / (len(param_values) + len(param_config['choices']))
                log_density += np.log(prob)

        return np.exp(log_density)

class RandomSampler:
    """عينات عشوائية"""
    def suggest(self, trials, search_space):
        return search_space.sample()

# ==============================================================================
# 3. نظام القطع المبكر (Pruning)
# ==============================================================================

class ASHAPruner:
    """Asynchronous Successive Halving Algorithm"""
    def __init__(self, min_resource=1, max_resource=100, reduction_factor=3):
        self.min_resource = min_resource
        self.max_resource = max_resource
        self.reduction_factor = reduction_factor

        # حساب المستويات
        self.rungs = []
        resource = max_resource
        while resource >= min_resource:
            self.rungs.append(resource)
            resource //= reduction_factor
        self.rungs = sorted(self.rungs)

        # تخزين النتائج
        self.rung_results = {rung: [] for rung in self.rungs}

    def should_prune(self, trial_id, step, value):
        """تحديد ما إذا كان يجب قطع التجربة"""
        # العثور على المستوى المناسب
        current_rung = None
        for rung in self.rungs:
            if step >= rung:
                current_rung = rung
                break

        if current_rung is None:
            return False

        # إضافة النتيجة
        self.rung_results[current_rung].append(value)

        # فحص القطع
        rung_values = self.rung_results[current_rung]
        if len(rung_values) >= self.reduction_factor:
            # حساب العتبة
            threshold_index = len(rung_values) // self.reduction_factor
            sorted_values = sorted(rung_values, reverse=True)
            threshold = sorted_values[threshold_index - 1] if threshold_index > 0 else sorted_values[0]

            return value < threshold

        return False

class MedianPruner:
    """قطع بناء على الوسيط"""
    def __init__(self, n_startup_trials=5, n_warmup_steps=10):
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps
        self.step_values = {}  # trial_id -> {step: value}

    def should_prune(self, trial_id, step, value, completed_trials):
        # حفظ القيمة
        if trial_id not in self.step_values:
            self.step_values[trial_id] = {}
        self.step_values[trial_id][step] = value

        # فحص الشروط
        if len(completed_trials) < self.n_startup_trials or step < self.n_warmup_steps:
            return False

        # جمع القيم من نفس الخطوة
        step_values = []
        for other_trial_id, values in self.step_values.items():
            if step in values:
                step_values.append(values[step])

        if len(step_values) < 2:
            return False

        # حساب الوسيط والقرار
        median_value = np.median(step_values)
        return value < median_value

# ==============================================================================
# 4. النظام الرئيسي
# ==============================================================================

class TrialObject:
    """كائن التجربة الذي يتم تمريره لدالة الهدف"""
    def __init__(self, trial_id, params, search_space):
        self.trial_id = trial_id
        self.params = params
        self.search_space = search_space

    def suggest_float(self, name, low, high, log=False):
        """اقتراح قيمة عشرية"""
        if name in self.params:
            return self.params[name]

        if log:
            value = np.exp(np.random.uniform(np.log(low), np.log(high)))
        else:
            value = np.random.uniform(low, high)

        self.params[name] = value
        return value

    def suggest_int(self, name, low, high):
        """اقتراح قيمة صحيحة"""
        if name in self.params:
            return self.params[name]

        value = np.random.randint(low, high + 1)
        self.params[name] = value
        return value

    def suggest_categorical(self, name, choices):
        """اقتراح قيمة فئوية"""
        if name in self.params:
            return self.params[name]

        value = np.random.choice(choices)
        self.params[name] = value
        return value

    def get(self, name, default=None):
        """الحصول على قيمة معامل"""
        return self.params.get(name, default)

    def __getitem__(self, key):
        """للوصول المباشر للمعاملات"""
        if key.startswith('suggest_'):
            return getattr(self, key)
        return self.params.get(key)

class HPOSystem:
    """النظام الرئيسي لضبط المعاملات الفائقة"""

    def __init__(self, search_space, objective_function, direction='maximize',
                 n_trials=100, n_startup_trials=10, sampler='TPE', pruner=None,
                 study_name='hpo_study', verbose=True):

        self.search_space = search_space
        self.objective_function = objective_function
        self.direction = direction
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials
        self.study_name = study_name
        self.verbose = verbose

        # إنشاء المكونات
        if sampler == 'TPE':
            self.sampler = TPESampler(n_startup_trials=n_startup_trials)
        else:
            self.sampler = RandomSampler()

        if pruner == 'ASHA':
            self.pruner = ASHAPruner()
        elif pruner == 'Median':
            self.pruner = MedianPruner()
        else:
            self.pruner = None

        # البيانات
        self.trials = []
        self.best_trial = None
        self.start_time = None

        # إحصائيات
        self.stats = {
            'n_complete': 0,
            'n_pruned': 0,
            'n_failed': 0,
            'total_time': 0
        }

        if self.verbose:
            print(f"✅ تم إنشاء نظام HPO: {study_name}")
            print(f"📊 فضاء البحث: {len(search_space.params)} معاملات")
            print(f"🎯 الاتجاه: {direction}")
            print(f"🔬 عدد التجارب: {n_trials}")

    def optimize(self):
        """تشغيل عملية التحسين"""
        self.start_time = time.time()

        if self.verbose:
            print(f"\n🚀 بداية التحسين...")

        for trial_num in range(self.n_trials):
            try:
                # اقتراح معاملات
                if trial_num < self.n_startup_trials:
                    params = self.search_space.sample()
                    method = 'عشوائي'
                else:
                    params = self.sampler.suggest(self.trials, self.search_space)
                    method = 'ذكي'

                # إنشاء trial object للدالة الهدف
                trial_id = f"trial_{trial_num:03d}"
                trial_obj = TrialObject(trial_id, params, self.search_space)

                # تشغيل التجربة
                start_time = time.time()

                try:
                    value = self.objective_function(trial_obj)
                    state = 'COMPLETE'
                except Exception as e:
                    if self.verbose:
                        print(f"❌ فشل في التجربة {trial_num}: {e}")
                    value = float('-inf') if self.direction == 'maximize' else float('inf')
                    state = 'FAILED'

                duration = time.time() - start_time

                # حفظ النتيجة
                trial_result = TrialResult(trial_id, params, value, state, duration)
                self.trials.append(trial_result)

                # تحديث أفضل تجربة
                self._update_best_trial(trial_result)

                # تحديث الإحصائيات
                self._update_stats()

                # طباعة النتيجة
                if self.verbose:
                    self._print_trial_result(trial_result, trial_num, method)

            except KeyboardInterrupt:
                if self.verbose:
                    print("\n⏹️ تم إيقاف التحسين بواسطة المستخدم")
                break
            except Exception as e:
                if self.verbose:
                    print(f"❌ خطأ في التجربة {trial_num}: {e}")
                continue

        # الانتهاء
        self.stats['total_time'] = time.time() - self.start_time

        if self.verbose:
            self.print_summary()

        return self.best_trial

    def _update_best_trial(self, trial):
        """تحديث أفضل تجربة"""
        if trial.state != 'COMPLETE':
            return

        is_better = False
        if self.best_trial is None:
            is_better = True
        elif self.direction == 'maximize':
            is_better = trial.value > self.best_trial.value
        else:
            is_better = trial.value < self.best_trial.value

        if is_better:
            self.best_trial = trial

    def _update_stats(self):
        """تحديث الإحصائيات"""
        self.stats['n_complete'] = sum(1 for t in self.trials if t.state == 'COMPLETE')
        self.stats['n_pruned'] = sum(1 for t in self.trials if t.state == 'PRUNED')
        self.stats['n_failed'] = sum(1 for t in self.trials if t.state == 'FAILED')

    def _print_trial_result(self, trial, trial_num, method):
        """طباعة نتيجة التجربة"""
        if trial.state == 'COMPLETE':
            is_best = trial == self.best_trial
            icon = "🎯" if is_best else "  "

            # تنسيق المعاملات للطباعة
            params_str = []
            for name, value in list(trial.params.items())[:3]:
                if isinstance(value, float):
                    if name == 'learning_rate':
                        params_str.append(f"{name}={value:.5f}")
                    else:
                        params_str.append(f"{name}={value:.3f}")
                else:
                    params_str.append(f"{name}={value}")

            params_display = " | ".join(params_str)
            if len(trial.params) > 3:
                params_display += "..."

            print(f"{icon} #{trial_num:3d} ({method:>6s}) | "
                  f"القيمة: {trial.value:.4f} | {params_display} | "
                  f"الوقت: {trial.duration:.1f}s")

    def print_summary(self):
        """طباعة ملخص النتائج"""
        print("\n" + "="*70)
        print("🏆 ملخص نتائج التحسين")
        print("="*70)

        if self.best_trial:
            print(f"🎯 أفضل قيمة: {self.best_trial.value:.6f}")
            print(f"🏅 أفضل تجربة: {self.best_trial.trial_id}")
            print(f"\n⚙️ أفضل معاملات:")
            for name, value in self.best_trial.params.items():
                if isinstance(value, float):
                    if name == 'learning_rate':
                        print(f"   {name}: {value:.6f}")
                    else:
                        print(f"   {name}: {value:.4f}")
                else:
                    print(f"   {name}: {value}")
        else:
            print("❌ لم يتم العثور على تجارب ناجحة")

        print(f"\n📊 إحصائيات:")
        print(f"   إجمالي التجارب: {len(self.trials)}")
        print(f"   تجارب مكتملة: {self.stats['n_complete']}")
        print(f"   تجارب فاشلة: {self.stats['n_failed']}")
        print(f"   الوقت الإجمالي: {self.stats['total_time']:.1f} ثانية")

        if self.stats['n_complete'] > 0:
            avg_time = sum(t.duration for t in self.trials if t.state == 'COMPLETE') / self.stats['n_complete']
            print(f"   متوسط وقت التجربة: {avg_time:.1f} ثانية")

        # أفضل 5 تجارب
        complete_trials = [t for t in self.trials if t.state == 'COMPLETE']
        if complete_trials:
            top_5 = sorted(complete_trials, key=lambda t: t.value, reverse=(self.direction == 'maximize'))[:5]
            print(f"\n🏆 أفضل 5 تجارب:")
            for i, trial in enumerate(top_5, 1):
                params_summary = []
                for name, value in list(trial.params.items())[:2]:
                    if isinstance(value, float):
                        params_summary.append(f"{name}={value:.3f}")
                    else:
                        params_summary.append(f"{name}={value}")

                params_str = " | ".join(params_summary)
                print(f"   {i}. {trial.value:.4f} | {params_str}")

        print(f"\n✅ انتهى التحسين!")

    def get_trials_dataframe(self):
        """تحويل النتائج إلى DataFrame"""
        if not self.trials:
            return pd.DataFrame()

        data = []
        for trial in self.trials:
            row = {
                'trial_id': trial.trial_id,
                'value': trial.value,
                'state': trial.state,
                'duration': trial.duration,
                **trial.params
            }
            data.append(row)

        return pd.DataFrame(data)

    def plot_optimization_history(self, save_path=None):
        """رسم تاريخ التحسين"""
        complete_trials = [t for t in self.trials if t.state == 'COMPLETE']
        if len(complete_trials) < 2:
            print("عدد التجارب المكتملة قليل جداً للرسم")
            return

        values = [t.value for t in complete_trials]
        trial_numbers = list(range(len(values)))

        # حساب أفضل قيمة تراكمية
        best_values = []
        current_best = values[0]
        for value in values:
            if self.direction == 'maximize':
                current_best = max(current_best, value)
            else:
                current_best = min(current_best, value)
            best_values.append(current_best)

        plt.figure(figsize=(12, 8))

        # الرسم الأساسي
        plt.subplot(2, 2, 1)
        plt.plot(trial_numbers, values, 'o-', alpha=0.6, markersize=4, label='قيم التجارب')
        plt.plot(trial_numbers, best_values, 'r-', linewidth=2, label='أفضل قيمة تراكمية')
        plt.xlabel('رقم التجربة')
        plt.ylabel('القيمة')
        plt.title('تاريخ التحسين')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # توزيع القيم
        plt.subplot(2, 2, 2)
        plt.hist(values, bins=min(15, len(values)//3), alpha=0.7, edgecolor='black')
        best_val = max(values) if self.direction == 'maximize' else min(values)
        plt.axvline(best_val, color='red', linestyle='--', linewidth=2, label='أفضل قيمة')
        plt.xlabel('القيمة')
        plt.ylabel('التكرار')
        plt.title('توزيع القيم')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # أوقات التجارب
        plt.subplot(2, 2, 3)
        durations = [t.duration for t in complete_trials]
        plt.plot(trial_numbers, durations, 'o-', alpha=0.7, color='green')
        plt.xlabel('رقم التجربة')
        plt.ylabel('الوقت (ثانية)')
        plt.title('مدة التجارب')
        plt.grid(True, alpha=0.3)

        # معدل التحسن
        plt.subplot(2, 2, 4)
        if len(best_values) > 1:
            improvements = []
            for i in range(1, len(best_values)):
                if self.direction == 'maximize':
                    improvement = best_values[i] - best_values[0]
                else:
                    improvement = best_values[0] - best_values[i]
                improvements.append(improvement)

            plt.plot(range(1, len(best_values)), improvements, 'o-', color='purple')
            plt.xlabel('رقم التجربة')
            plt.ylabel('التحسن من البداية')
            plt.title('معدل التحسن')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"تم حفظ الرسم في: {save_path}")

        plt.show()

# ==============================================================================
# 5. أمثلة جاهزة للتشغيل
# ==============================================================================

def example_mathematical():
    """مثال رياضي بسيط - البحث عن النقطة المثلى"""
    print("🔢 مثال رياضي: البحث عن أفضل قيم x, y")
    print("الهدف: تعظيم الدالة -(x-3)² - (y+2)² + 10")
    print("الحل المتوقع: x=3, y=-2, القيمة=10")

    def objective(trial):
        x = trial.suggest_float('x', -10, 10)
        y = trial.suggest_float('y', -10, 10)

        # محاكاة وقت الحوسبة
        time.sleep(0.05)

        # الدالة الهدف: نريد العثور على النقطة (3, -2)
        result = -(((x - 3)**2) + ((y + 2)**2)) + 10

        return result

    # إعداد فضاء البحث
    search_space = SearchSpace()
    search_space.add_uniform('x', -10, 10)
    search_space.add_uniform('y', -10, 10)

    # تشغيل التحسين
    hpo = HPOSystem(
        search_space=search_space,
        objective_function=objective,
        direction='maximize',
        n_trials=25,
        n_startup_trials=5,
        sampler='TPE',
        study_name='mathematical_example'
    )

    best_trial = hpo.optimize()

    # تحليل النتائج
    if best_trial:
        print(f"\n🎯 النتائج:")
        print(f"أفضل قيمة: {best_trial.value:.6f} (المتوقعة: 10.0)")
        print(f"أفضل x: {best_trial.params['x']:.6f} (المتوقعة: 3.0)")
        print(f"أفضل y: {best_trial.params['y']:.6f} (المتوقعة: -2.0)")

        # حساب الخطأ
        error_x = abs(best_trial.params['x'] - 3.0)
        error_y = abs(best_trial.params['y'] + 2.0)

        print(f"\nدقة النتائج:")
        print(f"خطأ x: {error_x:.6f}")
        print(f"خطأ y: {error_y:.6f}")

        if error_x < 0.1 and error_y < 0.1:
            print("✅ تم العثور على الحل الأمثل بدقة عالية!")
        elif error_x < 0.5 and error_y < 0.5:
            print("✅ تم العثور على حل قريب جداً من الأمثل!")
        else:
            print("⚠️ الحل يحتاج لمزيد من التجارب للوصول للدقة المطلوبة")

    return hpo

def example_ml_simulation():
    """مثال محاكاة تحسين نموذج تعلم آلة"""
    print("🤖 مثال تحسين نموذج تعلم آلة (محاكي)")
    print("الهدف: العثور على أفضل معاملات للحصول على أعلى دقة")

    def objective(trial):
        # المعاملات
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])
        hidden_size = trial.suggest_int('hidden_size', 64, 512)
        dropout = trial.suggest_float('dropout', 0.0, 0.8)
        optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])

        # محاكاة وقت التدريب (متغير حسب المعاملات)
        training_time = 0.1 + (hidden_size / 1000) + np.random.uniform(0, 0.3)
        time.sleep(training_time)

        # محاكاة الأداء بناء على المعاملات
        base_accuracy = 0.75

        # تأثير learning rate (أمثل حول 0.001-0.01)
        if 0.001 <= learning_rate <= 0.01:
            lr_bonus = 0.15
        elif 0.0001 <= learning_rate < 0.001:
            lr_bonus = 0.10
        elif 0.01 < learning_rate <= 0.05:
            lr_bonus = 0.08
        else:
            lr_bonus = 0.03

        # تأثير batch size
        batch_bonus = {16: 0.02, 32: 0.06, 64: 0.10, 128: 0.08, 256: 0.04}[batch_size]

        # تأثير hidden size (diminishing returns)
        hidden_bonus = min(0.10, (hidden_size - 64) / 400 * 0.10)

        # تأثير dropout (أمثل حول 0.2-0.4)
        if 0.2 <= dropout <= 0.4:
            dropout_bonus = 0.08
        elif 0.1 <= dropout < 0.2 or 0.4 < dropout <= 0.6:
            dropout_bonus = 0.04
        else:
            dropout_bonus = 0.01

        # تأثير optimizer
        optimizer_bonus = {'adam': 0.06, 'sgd': 0.04, 'rmsprop': 0.03}[optimizer]

        # تفاعلات بين المعاملات
        interaction_bonus = 0
        if optimizer == 'adam' and learning_rate <= 0.01:
            interaction_bonus += 0.03
        if batch_size >= 64 and hidden_size >= 128:
            interaction_bonus += 0.02
        if dropout > 0.1 and hidden_size > 256:
            interaction_bonus += 0.02

        # ضوضاء واقعية
        noise = np.random.normal(0, 0.04)

        # الدقة النهائية
        accuracy = base_accuracy + lr_bonus + batch_bonus + hidden_bonus + \
                  dropout_bonus + optimizer_bonus + interaction_bonus + noise

        # ضمان أن النتيجة في النطاق المعقول
        accuracy = max(0.5, min(0.99, accuracy))

        return accuracy

    # إعداد فضاء البحث
    search_space = SearchSpace()
    search_space.add_uniform('learning_rate', 1e-5, 1e-1, log=True)
    search_space.add_categorical('batch_size', [16, 32, 64, 128, 256])
    search_space.add_int('hidden_size', 64, 512)
    search_space.add_uniform('dropout', 0.0, 0.8)
    search_space.add_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])

    # تشغيل التحسين
    hpo = HPOSystem(
        search_space=search_space,
        objective_function=objective,
        direction='maximize',
        n_trials=30,
        n_startup_trials=8,
        sampler='TPE',
        study_name='ml_simulation_example'
    )

    best_trial = hpo.optimize()

    # تحليل النتائج
    if best_trial:
        print(f"\n🎯 النتائج:")
        print(f"أفضل دقة: {best_trial.value:.4f}")
        print(f"\n⚙️ أفضل معاملات:")
        for name, value in best_trial.params.items():
            if isinstance(value, float):
                if name == 'learning_rate':
                    print(f"   {name}: {value:.6f}")
                else:
                    print(f"   {name}: {value:.4f}")
            else:
                print(f"   {name}: {value}")

        # مقارنة مع القيم المثلى المتوقعة
        print(f"\n📊 تحليل النتائج:")
        lr = best_trial.params['learning_rate']
        if 0.001 <= lr <= 0.01:
            print("✅ Learning rate في النطاق الأمثل")
        else:
            print("⚠️ Learning rate قد يحتاج تحسين")

        if best_trial.params['optimizer'] == 'adam':
            print("✅ Adam optimizer اختيار ممتاز")

        if 64 <= best_trial.params['batch_size'] <= 128:
            print("✅ Batch size في النطاق الجيد")

    return hpo

def example_sklearn_real():
    """مثال حقيقي مع scikit-learn"""
    if not HAS_SKLEARN:
        print("❌ scikit-learn غير متوفر. تم تخطي المثال.")
        return None

    print("🔬 مثال حقيقي مع Scikit-learn")
    print("تحسين Random Forest على بيانات حقيقية")

    # تحضير البيانات
    try:
        # محاولة استخدام بيانات breast cancer
        data = load_breast_cancer()
        X, y = data.data, data.target
        print(f"البيانات: {X.shape[0]} عينة، {X.shape[1]} ميزة")
    except:
        # في حالة عدم توفرها، إنشاء بيانات صناعية
        X, y = make_classification(n_samples=1000, n_features=20,
                                  n_informative=15, n_redundant=5,
                                  n_classes=2, random_state=42)
        print(f"البيانات المصنعة: {X.shape[0]} عينة، {X.shape[1]} ميزة")

    def objective(trial):
        # معاملات Random Forest
        n_estimators = trial.suggest_int('n_estimators', 10, 200)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])

        # إنشاء النموذج
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=1  # لتجنب تعقيدات التوازي
        )

        # التقييم باستخدام cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 3 folds للسرعة
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

        return scores.mean()

    # إعداد فضاء البحث
    search_space = SearchSpace()
    search_space.add_int('n_estimators', 10, 200)
    search_space.add_int('max_depth', 3, 20)
    search_space.add_int('min_samples_split', 2, 20)
    search_space.add_int('min_samples_leaf', 1, 10)
    search_space.add_categorical('max_features', ['sqrt', 'log2'])

    # تشغيل التحسين
    hpo = HPOSystem(
        search_space=search_space,
        objective_function=objective,
        direction='maximize',
        n_trials=25,
        n_startup_trials=5,
        sampler='TPE',
        study_name='sklearn_example'
    )

    best_trial = hpo.optimize()

    # اختبار أفضل نموذج
    if best_trial:
        print(f"\n🎯 أفضل دقة: {best_trial.value:.4f}")
        print(f"🏆 أفضل معاملات:")
        for name, value in best_trial.params.items():
            print(f"   {name}: {value}")

        # إنشاء النموذج النهائي
        best_model = RandomForestClassifier(**best_trial.params, random_state=42)

        # تقييم نهائي
        final_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
        print(f"\n📊 التقييم النهائي (5-fold CV):")
        print(f"   المتوسط: {final_scores.mean():.4f}")
        print(f"   الانحراف المعياري: {final_scores.std():.4f}")
        print(f"   النطاق: [{final_scores.min():.4f}, {final_scores.max():.4f}]")

    return hpo

# ==============================================================================
# 6. نقطة الدخول الرئيسية
# ==============================================================================
if __name__ == '__main__':
    print("="*70)
    print("🌟 نظام HPO المتقدم - نسخة مكتملة وجاهزة للتشغيل 🌟")
    print("="*70)

    # مثال 1: رياضي
    print("\n\n--- [ المثال الأول: رياضي بسيط ] ---\n")
    hpo_math = example_mathematical()
    if hpo_math and hpo_math.stats['n_complete'] > 0:
        hpo_math.plot_optimization_history(save_path='hpo_math_history.png')

    # مثال 2: محاكاة تعلم آلة
    print("\n\n--- [ المثال الثاني: محاكاة تعلم آلة ] ---\n")
    hpo_sim = example_ml_simulation()
    if hpo_sim and hpo_sim.stats['n_complete'] > 0:
        hpo_sim.plot_optimization_history(save_path='hpo_ml_sim_history.png')

    # مثال 3: Scikit-learn حقيقي
    print("\n\n--- [ المثال الثالث: Scikit-learn حقيقي ] ---\n")
    hpo_sklearn = example_sklearn_real()
    if hpo_sklearn and hpo_sklearn.stats['n_complete'] > 0:
        hpo_sklearn.plot_optimization_history(save_path='hpo_sklearn_history.png')

    print("\n\n🎉 تم الانتهاء من جميع الأمثلة!")