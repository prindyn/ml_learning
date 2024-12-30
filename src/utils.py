import warnings
import matplotlib.pyplot as plt
import seaborn as sns


def age_cat(years):
    if years <= 20:
        return '0-20'
    elif years > 20 and years <= 30:
        return '20-30'
    elif years > 30 and years <= 40:
        return '30-40'
    elif years > 40 and years <= 50:
        return '40-50'
    elif years > 50 and years <= 60:
        return '50-60'
    elif years > 60 and years <= 70:
        return '60-70'
    elif years > 70:
        return '70+'


def bi_cat_countplot(df, column, hue_column):
    unique_hue_values = df[hue_column].unique()
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(14, 6)

    pltname = f'Нормалізований розподіл значень за категорією: {column}'
    proportions = df.groupby(hue_column)[column].value_counts(normalize=True)
    proportions = (proportions*100).round(2)
    ax = proportions.unstack(hue_column).sort_values(
        by=unique_hue_values[0], ascending=False
    ).plot.bar(ax=axes[0], title=pltname)

    # анотація значень в барплоті
    for container in ax.containers:
        ax.bar_label(container, fmt='{:,.1f}%')

    pltname = f'Кількість даних за категорією: {column}'
    counts = df.groupby(hue_column)[column].value_counts()
    ax = counts.unstack(hue_column).sort_values(
        by=unique_hue_values[0], ascending=False
    ).plot.bar(ax=axes[1], title=pltname)

    for container in ax.containers:
        ax.bar_label(container)


def uni_cat_target_compare(df, column):
    bi_cat_countplot(df, column, hue_column='TARGET')


def bi_countplot_target(df0, df1, column, hue_column):
    pltname = 'Клієнт зі складнощами щодо платності'
    print(pltname.upper())
    bi_cat_countplot(df1, column, hue_column)
    plt.show()

    pltname = 'Клієнти зі своєчасними платежами'
    print(pltname.upper())
    bi_cat_countplot(df0, column, hue_column)
    plt.show()


def outlier_range(dataset, column):
    Q1 = dataset[column].quantile(0.25)
    Q3 = dataset[column].quantile(0.75)
    IQR = Q3 - Q1
    Min_value = (Q1 - 1.5 * IQR)
    Max_value = (Q3 + 1.5 * IQR)
    return Max_value


def dist_box(dataset, column):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        plt.figure(figsize=(16, 6))

        plt.subplot(1, 2, 1)
        sns.histplot(dataset[column], color='purple', kde=True)
        pltname = 'Графік розподілу для ' + column
        plt.ticklabel_format(style='plain', axis='x')
        plt.title(pltname)

        plt.subplot(1, 2, 2)
        red_diamond = dict(markerfacecolor='r', marker='D')
        sns.boxplot(y=column, data=dataset, flierprops=red_diamond)
        pltname = 'Боксплот для ' + column
        plt.title(pltname)

        plt.show()


def dist_box_filter(df, column_name, filter_value):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.subplots(1, 2, figsize=(20, 8))

        # Distribution plot
        plt.subplot(121)
        sns.distplot(df[df[column_name] < filter_value][column_name])
        pltname = f'Distplot of {column_name}'
        plt.title(pltname)

        # Boxplot
        plt.subplot(122)
        sns.boxplot(df[df[column_name] < filter_value][column_name])
        pltname = f'Boxplot of {column_name}'
        plt.title(pltname)

        # Layout adjustment
        plt.tight_layout(pad=4)
        plt.show()


def kde_no_outliers(df0, df1, Max_value0, Max_value1, column):
    plt.figure(figsize=(14, 6))
    sns.kdeplot(df1[df1[column] <= Max_value1]
                [column], label='Payment difficulties')
    sns.kdeplot(df0[df0[column] <= Max_value0]
                [column], label='On-Time Payments')
    plt.ticklabel_format(style='plain', axis='x')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()


def corr_payment_scatter(df1, df0, max_value0_0, max_value0_1, max_value1_0, max_value1_1, column0, column1):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.title('Payment difficulties')
    sns.scatterplot(x=df1[df1[column0] < max_value1_0][column0],
                    y=df1[df1[column1] < max_value1_1][column1], data=df1)
    plt.ticklabel_format(style='plain', axis='x')
    plt.ticklabel_format(style='plain', axis='y')

    plt.subplot(1, 2, 2)
    plt.title('On-Time Payments')
    sns.scatterplot(x=df0[df0[column0] < max_value0_0][column0],
                    y=df0[df0[column1] < max_value0_1][column1], data=df0)
    plt.ticklabel_format(style='plain', axis='x')
    plt.ticklabel_format(style='plain', axis='y')

    plt.tight_layout(pad=4)
    plt.show()


def draw_boxplot(df, categorical, continuous, max_continuous, title, hue_column, subplot_position):
    """
    Малює блок-діаграму для заданого DataFrame, категоріальної та неперервної змінної.
    """
    plt.subplot(1, 2, subplot_position)
    plt.title(title)
    red_diamond = dict(markerfacecolor='r', marker='D')
    sns.boxplot(x=categorical,
                y=df[df[continuous] < max_continuous][continuous],
                data=df,
                flierprops=red_diamond,
                order=sorted(df[categorical].unique(), reverse=True),
                hue=hue_column, hue_order=sorted(df[hue_column].unique(), reverse=True))
    plt.ticklabel_format(style='plain', axis='y')
    plt.xticks(rotation=90)


def bi_boxplot(df1, df0, categorical, continuous, max_continuous1, max_continuous0, hue_column):
    """
    Створює паралельні блок-діаграми для двох груп, визначених у наборі даних, на основі
    категоріальної та неперервної змінної, виділяючи відмінності за допомогою відтінків.
    """
    plt.figure(figsize=(16, 10))

    # Графік для першо групи "Труднощі з платежами" (Payment Difficulties)
    draw_boxplot(df1, categorical, continuous, max_continuous1,
                 'Payment Difficulties', hue_column, 1)

    # Графік для другої групи "Вчасні оплати" (On-Time Payments)
    draw_boxplot(df0, categorical, continuous, max_continuous0,
                 'On-Time Payments', hue_column, 2)

    plt.tight_layout(pad=4)
    plt.show()


def numeric_vs_categorical_analysis(df0, df1, column_1, column_2, column_3):
    max_value1_column_1 = outlier_range(df1, column_1)
    max_value0_column_1 = outlier_range(df0, column_1)

    # Клієнт з платіжними труднощами
    df1.groupby(by=[column_2, column_3])[column_1].describe().head()

    # Клієнт зі своєчасними платежами
    df0.groupby(by=[column_2, column_3])[column_1].describe().head()

    bi_boxplot(column_2, column_1, max_value1_column_1,
               max_value0_column_1, column_3)


def bi_countplot_target(df0, df1, column, hue_column):
    group_name = f'Нормалізований розподіл значень за категорією: {column}'
    print(group_name.upper())

    pltname = 'Клієнт зі складнощями щодо платності'
    unique_hue_values = df1[hue_column].unique()
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(14, 4)

    proportions = df1.groupby(hue_column)[column].value_counts(normalize=True)
    proportions = (proportions*100).round(2)
    ax = proportions.unstack(hue_column).sort_values(
        by=unique_hue_values[0], ascending=False
    ).plot.bar(ax=axes[0], title=pltname)

    # анотація значень в барплоті
    for container in ax.containers:
        ax. bar_label(container, fmt='{:,.1f}%')

    pltname = 'Клієнти зі своєчасними платежами'
    unique_hue_values = df0[hue_column].unique()

    proportions = df0.groupby(hue_column)[column].value_counts(normalize=True)
    proportions = (proportions*100).round(2)
    ax = proportions.unstack(hue_column).sort_values(
        by=unique_hue_values[0], ascending=False
    ).plot.bar(ax=axes[1], title=pltname)

    for container in ax.containers:
        ax.bar_label(container, fmt='{:,.1f}%')

    plt.show()

    # ------------
    group_name = f'Кількість значень за категорією {column}'
    print(group_name.upper())

    pltname = 'Клієнт зі складнощями щодо платності'
    unique_hue_values = df1[hue_column].unique()
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(14, 4)
    counts = df1.groupby(hue_column)[column].value_counts()
    ax = counts.unstack(hue_column).sort_values(
        by=unique_hue_values[0], ascending=False
    ).plot.bar(ax=axes[0], title=pltname)

    for container in ax.containers:
        ax.bar_label(container)

    pltname = 'Клієнти зі своєчасними платежами'
    unique_hue_values = df0[hue_column].unique()
    counts = df0.groupby(hue_column)[column].value_counts()
    ax = counts.unstack(hue_column).sort_values(
        by=unique_hue_values[0], ascending=False
    ).plot.bar(ax=axes[1], title=pltname)

    for container in ax. containers:
        ax.bar_label(container)

    plt.show()
