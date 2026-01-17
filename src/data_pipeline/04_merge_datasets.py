"""
04_merge_datasets.py
دمج جميع مجموعات البيانات في مجموعة بيانات رئيسية واحدة
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def load_all_datasets():
    """
    تحميل جميع مجموعات البيانات المتاحة
    """
    print("جاري تحميل مجموعات البيانات...")
    
    datasets = {}
    
    # 1. تحميل المواقع المخصبة
    enhanced_path = 'data/processed/candidates_enhanced.csv'
    if os.path.exists(enhanced_path):
        datasets['sites'] = pd.read_csv(enhanced_path)
        print(f"  ✓ تم تحميل {len(datasets['sites'])} موقع")
    else:
        print("  ✗ ملف المواقع المخصبة غير موجود")
        return None
    
    # 2. تحميل بيانات الطقس
    weather_path = 'data/processed/weather_summary.csv'
    if os.path.exists(weather_path):
        datasets['weather'] = pd.read_csv(weather_path)
        print(f"  ✓ تم تحميل بيانات الطقس لـ {len(datasets['weather'])} موقع")
    else:
        datasets['weather'] = None
        print("  ⚠️ بيانات الطقس غير متوفرة")
    
    # 3. تحميل مصفوفة المسافات
    distance_path = 'data/processed/distance_matrix.npy'
    if os.path.exists(distance_path):
        datasets['distances'] = np.load(distance_path)
        print(f"  ✓ تم تحميل مصفوفة المسافات: {datasets['distances'].shape}")
    else:
        datasets['distances'] = None
        print("  ⚠️ مصفوفة المسافات غير متوفرة")
    
    return datasets

def merge_datasets(datasets):
    """
    دمج جميع البيانات في مجموعة بيانات رئيسية
    """
    print("\nجاري دمج مجموعات البيانات...")
    
    master_df = datasets['sites'].copy()
    
    # دمج بيانات الطقس إذا كانت متوفرة
    if datasets['weather'] is not None:
        master_df = pd.merge(master_df, datasets['weather'], 
                           on='site_id', how='left')
    else:
        # إضافة قيم افتراضية للطقس
        master_df['weather_score'] = 0.7
        master_df['temp_mean'] = 20.0
        master_df['wind_max'] = 6.0
    
    # حساب النتيجة الأولية
    master_df = calculate_preliminary_scores(master_df)
    
    # إضافة معلومات المصفوفة
    if datasets['distances'] is not None:
        master_df = add_distance_info(master_df, datasets['distances'])
    
    # إضافة أعمدة إضافية
    master_df = add_derived_features(master_df)
    
    # ترتيب الأعمدة
    master_df = organize_columns(master_df)
    
    return master_df

def calculate_preliminary_scores(df):
    """
    حساب النتائج الأولية باستخدام أوزان افتراضية
    """
    # الأوزان الافتراضية
    WEIGHTS = {
        'risk': 0.35,
        'access': 0.25,
        'priority': 0.30,
        'weather': 0.10
    }
    
    # حساب النتيجة الأساسية
    df['base_score'] = (
        WEIGHTS['risk'] * (1 - df['base_risk_score']) +
        WEIGHTS['access'] * df['base_access_score'] +
        WEIGHTS['priority'] * df['base_priority_score']
    ).round(3)
    
    # إضافة تأثير الطقس إذا كان متوفراً
    if 'weather_score' in df.columns:
        df['total_score'] = (
            df['base_score'] * 0.9 + df['weather_score'] * 0.1
        ).round(3)
    else:
        df['total_score'] = df['base_score']
    
    # إضافة رتبة بناء على النتيجة
    df['rank'] = df['total_score'].rank(ascending=False, method='min').astype(int)
    
    return df

def add_distance_info(df, distance_matrix):
    """
    إضافة معلومات المسافات لكل موقع
    """
    n = len(df)
    
    # حساب متوسط المسافة لأقرب 5 مواقع
    nearest_distances = []
    for i in range(n):
        # الحصول على جميع المسافات للموقع i
        distances = distance_matrix[i]
        
        # استبعاد المسافة إلى النفس (صفر)
        other_distances = distances[distances > 0]
        
        if len(other_distances) >= 5:
            # أقرب 5 مسافات
            nearest_5 = np.sort(other_distances)[:5]
            avg_nearest = np.mean(nearest_5)
            min_distance = nearest_5[0]
        else:
            avg_nearest = np.mean(other_distances) if len(other_distances) > 0 else 10.0
            min_distance = np.min(other_distances) if len(other_distances) > 0 else 10.0
        
        nearest_distances.append({
            'avg_distance_to_5_nearest_km': round(avg_nearest, 2),
            'min_distance_to_nearest_km': round(min_distance, 2)
        })
    
    distance_df = pd.DataFrame(nearest_distances)
    df = pd.concat([df, distance_df], axis=1)
    
    return df

def add_derived_features(df):
    """
    إضافة خصائص مشتقة
    """
    # كفاءة التكلفة (عدد الأشخاص لكل دولار)
    df['cost_efficiency'] = (
        df['population_served'] / df['estimated_cost_usd']
    ).round(6)
    
    # نسبة المخاطرة (أولوية عالية + خطر عالي)
    df['risk_priority_ratio'] = (
        df['base_priority_score'] / (df['base_risk_score'] + 0.01)
    ).round(3)
    
    # مؤشر الجاهزية
    df['readiness_index'] = (
        df['base_access_score'] * 0.4 +
        df['weather_score'] * 0.3 +
        (1 - df['base_risk_score']) * 0.3
    ).round(3)
    
    # فئة النتيجة
    df['score_category'] = pd.cut(
        df['total_score'],
        bins=[0, 0.4, 0.6, 0.8, 1.0],
        labels=['ضعيف', 'متوسط', 'جيد', 'ممتاز']
    )
    
    # إضافة الطابع الزمني
    df['last_processed'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return df

def organize_columns(df):
    """
    ترتيب الأعمدة بشكل منطقي
    """
    column_order = [
        # المعرفات الأساسية
        'site_id', 'name_ar', 'name_en', 'region', 'zone', 'site_type',
        
        # الموقع الجغرافي
        'latitude', 'longitude',
        
        # القيم الأساسية
        'base_risk_score', 'risk_category',
        'base_access_score', 'access_category',
        'base_priority_score',
        
        # البيانات المناخية
        'weather_score', 'temp_mean', 'wind_max', 'humidity_mean',
        
        # النتائج المحسوبة
        'base_score', 'total_score', 'rank', 'score_category',
        
        # مؤشرات إضافية
        'urgency_index', 'readiness_index', 'risk_priority_ratio',
        
        # معلومات المسافة
        'avg_distance_to_5_nearest_km', 'min_distance_to_nearest_km',
        
        # المعلومات الديموغرافية والمالية
        'population_served', 'estimated_cost_usd', 'cost_efficiency',
        
        # الأعلام والتصنيفات
        'is_critical', 'needs_special_access',
        
        # البيانات الوصفية
        'data_source', 'creation_date', 'last_processed'
    ]
    
    # الاحتفاظ فقط بالأعمدة الموجودة
    existing_columns = [col for col in column_order if col in df.columns]
    additional_columns = [col for col in df.columns if col not in existing_columns]
    
    final_columns = existing_columns + additional_columns
    df = df[final_columns]
    
    return df

def save_master_dataset(df, output_path='data/processed/master_dataset.csv'):
    """
    حفظ مجموعة البيانات الرئيسية
    """
    # تأكد من وجود المجلد
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # حفظ كـ CSV
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    # حفظ نسخة إضافية كـ Excel (اختياري)
    excel_path = output_path.replace('.csv', '.xlsx')
    df.to_excel(excel_path, index=False)
    
    print(f"\n✓ تم حفظ مجموعة البيانات الرئيسية:")
    print(f"  - CSV: {output_path}")
    print(f"  - Excel: {excel_path}")
    print(f"  - عدد السجلات: {len(df)}")
    print(f"  - عدد الأعمدة: {len(df.columns)}")
    
    return df

def main():
    """
    الدالة الرئيسية
    """
    print("=" * 60)
    print("دمج مجموعات البيانات - Gaza Humanitarian Sites")
    print("=" * 60)
    
    # تحميل جميع البيانات
    datasets = load_all_datasets()
    if datasets is None:
        print("فشل في تحميل البيانات. تأكد من تشغيل الملفات السابقة.")
        return
    
    # دمج البيانات
    master_df = merge_datasets(datasets)
    
    # حفظ النتيجة
    master_df = save_master_dataset(master_df)
    
    # عرض ملخص
    print("\n" + "=" * 60)
    print("ملخص مجموعة البيانات الرئيسية:")
    print("-" * 60)
    print(f"أفضل 5 مواقع بناء على النتيجة الإجمالية:")
    top_5 = master_df.nlargest(5, 'total_score')[['site_id', 'name_ar', 'total_score', 'region', 'site_type']]
    print(top_5.to_string(index=False))
    
    print(f"\nالتوزيع حسب المنطقة:")
    region_dist = master_df['region'].value_counts()
    for region, count in region_dist.items():
        print(f"  {region}: {count} موقع")
    
    print(f"\nالتوزيع حسب فئة النتيجة:")
    score_dist = master_df['score_category'].value_counts()
    for category, count in score_dist.items():
        print(f"  {category}: {count} موقع")
    
    print(f"\nنطاق النتائج: {master_df['total_score'].min():.3f} - {master_df['total_score'].max():.3f}")
    print(f"متوسط النتيجة: {master_df['total_score'].mean():.3f}")
    
    print("\n" + "=" * 60)
    print("تم الانتهاء من دمج البيانات بنجاح!")
    print("=" * 60)
    
    return master_df

if __name__ == "__main__":
    main()
