# تقرير تحليل المبيعات والتعلم الآلي (ML Sales Analysis Report)

هذا التقرير يوضح الخطوات التفصيلية لتحليل مجموعة بيانات المبيعات الخاصة بك وتطبيق نماذج التعلم الآلي للتنبؤ بإمكانات المبيعات الجغرافية وإجمالي الإيرادات للعام القادم.

---

## 1. إعداد البيانات والتنظيف الأولي (Data Preparation and Cleaning)

**الهدف:** تحميل البيانات، تحويل الأعمدة الأساسية إلى التنسيق الصحيح، واختيار الأعمدة الضرورية للتحليل.

```python
import pandas as pd
import numpy as np

# 1. تحميل البيانات
# افترض أن ملف البيانات الخاص بك هو 'your_sales_data_file.csv'
# يجب استبدال هذا المسار بالمسار الفعلي لملفك
try:
    df = pd.read_csv('your_sales_data_file.csv')
except FileNotFoundError:
    print("خطأ: لم يتم العثور على ملف البيانات. يرجى التحقق من المسار.")
    exit()

# 2. تحديد الأعمدة الأساسية للتحليل
essential_columns = [
    'order_id', 
    'order_date', 
    'total', 
    'qty_ordered', 
    'City', 
    'State', 
    'Region',
    'category' 
]

# إنشاء إطار بيانات جديد يحتوي على الأعمدة الأساسية فقط
df_ml = df[essential_columns].copy()

# 3. تحويل الأنواع وتنظيف البيانات
# تحويل 'order_date' إلى تنسيق التاريخ والوقت
df_ml['order_date'] = pd.to_datetime(df_ml['order_date'], errors='coerce')

# التأكد من أن الأعمدة الرقمية هي من النوع الصحيح وحذف الصفوف الفارغة
df_ml['total'] = pd.to_numeric(df_ml['total'], errors='coerce')
df_ml['qty_ordered'] = pd.to_numeric(df_ml['qty_ordered'], errors='coerce')
df_ml.dropna(subset=['order_date', 'total', 'qty_ordered'], inplace=True)

# حفظ إطار البيانات المُركز عليه للمراحل التالية
df_ml.to_csv('focused_sales_data.csv', index=False)

print("تم تنظيف البيانات واختيار الأعمدة الأساسية بنجاح.")
print(df_ml.head())
```

**الشرح:**
*   تم تحميل مكتبة `pandas` للتعامل مع البيانات.
*   تم تحديد قائمة بـ **الأعمدة الأساسية** اللازمة للتنبؤ (التاريخ، الإيرادات، الكمية، الموقع، الفئة).
*   تم إنشاء إطار بيانات جديد (`df_ml`) يحتوي على هذه الأعمدة فقط، مما يضمن **وضوح** و **كفاءة** التحليل.
*   تم تحويل عمود `order_date` إلى تنسيق `datetime`، وهو أمر ضروري لتحليل السلاسل الزمنية.

---

## 2. تحليل البيانات الاستكشافي (Exploratory Data Analysis - EDA)

**الهدف:** فهم خصائص البيانات وتوزيعها وعلاقاتها من خلال الإحصائيات والتصور.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 1. الإحصائيات الوصفية
print("\n--- الإحصائيات الوصفية للإيرادات والكمية ---")
print(df_ml[['total', 'qty_ordered']].describe())

# 2. توزيع الإيرادات (Revenue Distribution)
plt.figure(figsize=(10, 6))
sns.histplot(df_ml['total'], bins=50, kde=True)
plt.title('توزيع إجمالي الإيرادات')
plt.xlabel('إجمالي الإيرادات')
plt.ylabel('التكرار')
plt.show()
# plt.savefig('revenue_distribution.png') # لحفظ الرسم البياني

# 3. اتجاه المبيعات عبر الزمن (Sales Trend Over Time)
# تجميع البيانات على أساس يومي
daily_sales = df_ml.set_index('order_date')['total'].resample('D').sum().fillna(0)

plt.figure(figsize=(14, 7))
daily_sales.plot()
plt.title('اتجاه إجمالي الإيرادات اليومية')
plt.xlabel('التاريخ')
plt.ylabel('إجمالي الإيرادات')
plt.show()
# plt.savefig('daily_revenue_trend.png') # لحفظ الرسم البياني

# 4. تحليل الفئات الأكثر ربحية
category_revenue = df_ml.groupby('category')['total'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=category_revenue.index, y=category_revenue.values)
plt.title('أعلى 10 فئات منتجات حسب إجمالي الإيرادات')
plt.xlabel('فئة المنتج')
plt.ylabel('إجمالي الإيرادات')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
# plt.savefig('top_categories_revenue.png') # لحفظ الرسم البياني
```

**الشرح:**
*   تم استخدام `describe()` للحصول على ملخص إحصائي سريع.
*   تم إنشاء رسوم بيانية (Histograms) لتصور **توزيع الإيرادات**، مما يكشف عن القيم المتطرفة (Outliers) ودرجة الانحراف.
*   تم استخدام `resample('D')` لرسم **اتجاه المبيعات اليومية**، وهو أمر حيوي لفهم الموسمية والأنماط الزمنية.
*   تم تحليل **أداء فئات المنتجات** لتحديد الفئات التي تساهم بأكبر قدر في الإيرادات.

---

## 3. التنبؤ بإمكانات المبيعات الجغرافية (Geographical Sales Potential) - التصنيف

**الهدف:** استخدام التجميع العنقودي (Clustering) لتعريف المدن ذات القيمة العالية، ثم استخدام نماذج التصنيف للتنبؤ بها.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

# 1. تجميع البيانات حسب الموقع وإنشاء ميزات المبيعات
location_sales = df_ml.groupby(['State', 'City']).agg(
    Total_Revenue=('total', 'sum'),
    Total_Orders=('order_id', 'nunique'),
    Avg_Order_Value=('total', 'mean')
).reset_index()

location_sales['Revenue_Per_Order'] = location_sales['Total_Revenue'] / location_sales['Total_Orders']

# 2. التجميع العنقودي (K-Means) لتحديد المجموعات
X_cluster = location_sales[['Total_Revenue', 'Total_Orders', 'Avg_Order_Value', 'Revenue_Per_Order']].copy()
# تطبيق تحويل لوغاريتمي وتوحيد قياسي للبيانات قبل التجميع
X_cluster = np.log1p(X_cluster)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
location_sales['Cluster'] = kmeans.fit_predict(X_scaled)

# 3. تعريف المتغير الهدف (High_Value_City)
# تحديد المجموعة ذات أعلى متوسط إيرادات كـ "قيمة عالية" (1)
cluster_summary = location_sales.groupby('Cluster')['Total_Revenue'].mean().sort_values(ascending=False)
high_value_cluster = cluster_summary.index[0]
location_sales['High_Value_City'] = (location_sales['Cluster'] == high_value_cluster).astype(int)

# 4. مقارنة نماذج التصنيف
X = location_sales[['Total_Revenue', 'Total_Orders', 'Avg_Order_Value', 'Revenue_Per_Order']]
y = location_sales['High_Value_City']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

models = {
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Support Vector Machine': SVC(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append({'Model': name, 'Accuracy': f'{accuracy:.4f}', 'F1-Score': f'{f1:.4f}'})

comparison_df = pd.DataFrame(results)
print("\n--- جدول مقارنة النماذج (إمكانات المبيعات الجغرافية) ---")
print(comparison_df.to_markdown(index=False))
```

**الشرح:**
*   تم تجميع البيانات حسب الموقع لإنشاء مقاييس مبيعات لكل مدينة.
*   تم استخدام **K-Means** لتقسيم المدن إلى مجموعات، مما يسمح بتعريف المدن ذات الأداء العالي.
*   تم إنشاء متغير هدف ثنائي (0 أو 1) بناءً على هذه المجموعات.
*   تم تدريب ومقارنة 7 نماذج تصنيف باستخدام مقاييس **الدقة** و **مقياس F1** لتحديد النموذج الأفضل للتنبؤ بما إذا كانت مدينة جديدة ستكون ذات قيمة عالية.

---

## 4. التنبؤ بإجمالي الإيرادات للعام القادم (Total Revenue Forecasting) - الانحدار

**الهدف:** تطبيق نماذج الانحدار على بيانات السلاسل الزمنية للتنبؤ بإجمالي الإيرادات للعام القادم.

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. هندسة ميزات السلاسل الزمنية
daily_sales = df_ml.set_index('order_date')['total'].resample('D').sum().fillna(0).reset_index()
daily_sales.columns = ['Date', 'Revenue']

daily_sales['Year'] = daily_sales['Date'].dt.year
daily_sales['Month'] = daily_sales['Date'].dt.month
daily_sales['DayOfWeek'] = daily_sales['Date'].dt.dayofweek

# إنشاء ميزات التأخير (Lag Features)
for i in range(1, 4):
    daily_sales[f'Lag_{i}'] = daily_sales['Revenue'].shift(i)

daily_sales.dropna(inplace=True)

# 2. مقارنة نماذج الانحدار
X = daily_sales.drop(['Date', 'Revenue'], axis=1)
y = daily_sales['Revenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

regression_models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'XGBoost Regressor': XGBRegressor(random_state=42, objective='reg:squarederror')
}

regression_results = []
for name, model in regression_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    regression_results.append({'Model': name, 'Mean Squared Error (MSE)': f'{mse:.2f}', 'R-squared (R2)': f'{r2:.4f}'})

regression_comparison_df = pd.DataFrame(regression_results)
print("\n--- جدول مقارنة النماذج (التنبؤ بالإيرادات) ---")
print(regression_comparison_df.to_markdown(index=False))

# 3. التنبؤ بإجمالي الإيرادات للعام القادم (باستخدام أفضل نموذج - XGBoost كمثال)
best_model = XGBRegressor(random_state=42, objective='reg:squarederror')
best_model.fit(X, y)

last_date = daily_sales['Date'].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=365, freq='D')
future_df = pd.DataFrame({'Date': future_dates})

# إنشاء ميزات التأخير للتواريخ المستقبلية (تبسيط للتوضيح)
last_3_revenues = daily_sales['Revenue'].tail(3).values
future_df['Lag_1'] = [last_3_revenues[2]] + [0] * 364
future_df['Lag_2'] = [last_3_revenues[1]] + [0] * 364
future_df['Lag_3'] = [last_3_revenues[0]] + [0] * 364

# إنشاء ميزات الوقت للتواريخ المستقبلية
future_df['Year'] = future_df['Date'].dt.year
future_df['Month'] = future_df['Date'].dt.month
future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek

X_future = future_df.drop('Date', axis=1)
future_df['Predicted_Revenue'] = best_model.predict(X_future)

total_predicted_revenue = future_df['Predicted_Revenue'].sum()

print(f"\n--- إجمالي الإيرادات المتوقعة للعام القادم ---")
print(f"الإيرادات المتوقعة: ${total_predicted_revenue:,.2f}")
```

**الشرح:**
*   تم تجميع البيانات يوميًا لإنشاء سلسلة زمنية للإيرادات.
*   تم إنشاء ميزات زمنية (مثل اليوم من الأسبوع والشهر) وميزات التأخير (Lag Features) التي تساعد النموذج على فهم الأنماط المتكررة.
*   تم تدريب ومقارنة نماذج الانحدار باستخدام مقاييس **MSE** و **R2**.
*   تم استخدام النموذج الأفضل للتنبؤ بالإيرادات اليومية للـ 365 يومًا القادمة، وتم جمعها للحصول على **إجمالي الإيرادات المتوقعة للعام القادم**.
