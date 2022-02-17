import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import lines
from matplotlib.font_manager import FontProperties
import matplotlib
import re
import explanation_utils
import matplotlib.patches as mpatches

# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams['font.sans-serif'] = "Arial"
#matplotlib.rcParams['font.sans-serif'] = "Consolas"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "sans-serif"
#matplotlib.rcParams['font.family'] = "monospace"


def insert_space(text):
    # find text before '=' sign
    elements = re.findall('[\w]=', text)
    if len(elements) > 0:
        for e in elements:
            text = text.replace(e, e.replace('=', ' ='))

    # find text after '=' sign
    elements = re.findall('=[\w]', text)
    if len(elements) > 0:
        for e in elements:
            text = text.replace(e, e.replace('=', '= '))
    return text


def probability_to_text(class_pred, class_proba):
    """
    if class_proba < 0.66:
        text = 'likely to be ' + class_pred
    elif class_proba >= 0.66 and class_proba < 0.83:
        text = 'very likely to be ' + class_pred
    elif class_proba >= 0.83:
        text = 'extremely likely to be ' + class_pred
    """
    if class_proba < 0.75 and class_proba > 0.25:
        text = 'with moderate certainty'
    elif class_proba >= 0.75 or class_proba <= 0.25:
        text = 'with high certainty'
    else:
        text = 'error'

    return text

    if rename:
        data = data.rename(columns=explanation_utils.ExplanationUtils.rename_dict)
    ordered_attributes = ['Amount (EUR)', 'Duration (months)', 'Purpose', 'Account Balance',
                          'Assets', 'Available Income', 'Housing', 'Loan History',
                          'Number of Previous Loans', 'Other Loans',
                          'Savings Account', 'Age (years)', 'Employment', 'Foreign Worker',
                          'Job', 'Personal Status / Sex', 'Residence Duration', 'Telephone']
    vals = data[ordered_attributes].iloc[idx].values
    # Replace .0 with empty string to convert to int
    vals = [re.sub(r'(\.0)0*', '', str(val)) for val in vals]
    cols = data[ordered_attributes].columns
    df = pd.DataFrame({'Attribute': cols, 'Value': vals})

    return df

def reshape_instance(data, idx, rename=False):
    #print(data)
    if rename:
        data = data.rename(columns=explanation_utils.ExplanationUtils.rename_dict)
    ordered_attributes = ['Amount (EUR)', 'Duration (months)', 'Guarantee', 'Purpose', 'Account Balance',
                          'Assets', 'Available Income', 'Housing', 'Loan History', 'Number of Previous Loans', 'Other Loans',
                          'Savings Account', 'Age (years)', 'Employment', 'Job', 'Number of dependents', 'Residence Duration', 'Telephone']
    vals = data[ordered_attributes].loc[idx].values
    # Replace .0 with empty string to convert to int
    vals = [re.sub(r'(\.0)0*', '', str(val)) for val in vals]
    cols = data[ordered_attributes].columns
    df = pd.DataFrame({'Attribute': cols, 'Value': vals})

    return df




def reshape_dice_exp(exp, version, fill_empty):
    # Order of the attributes
    ordered_attributes = ['Amount (EUR)', 'Duration (months)', 'Guarantee', 'Purpose', 'Account Balance',
                          'Assets', 'Available Income', 'Housing', 'Loan History', 'Number of Previous Loans', 'Other Loans',
                          'Savings Account', 'Age (years)', 'Employment', 'Job', 'Number of dependents', 'Residence Duration', 'Telephone']
    # Original instance of the explanation as list
    orig_instance_df = exp.test_instance_df if version == '0.5' else exp.org_instance
    orig_instance = orig_instance_df[ordered_attributes].iloc[0].to_list()
    # Create df of Counterfactuals where unchanged attributes are '-'
    CFs = {}
    dice_exp_df = exp.final_cfs_df_sparse[ordered_attributes].copy()
    dice_exp_df.loc[:, 'Amount (EUR)'] = dice_exp_df['Amount (EUR)'].round(0)  # Round amount to 0 decimals
    dice_exp_list = [list(exp) for exp in dice_exp_df.values]

    for i, cf in enumerate(dice_exp_list):
        cf_list_compare = list(zip(cf, orig_instance))
        change_list = [str(val_change) if val_change != val_org else fill_empty for (val_change, val_org) in cf_list_compare]
        CFs['CF_' + str(i)] = change_list

    # Replace .0 with empty string to convert to int
    return pd.DataFrame(CFs).replace(regex=r'(\.0)0*', value='')



def dice_plot(data, idx, exp, save_to, rename=False, version='0.4', fill_empty='-', fill_cell=False, alignment='right'):
    """
    Creates and saves plot of CF explanation
    :param data: Input data
    :param idx: Index of instance of interest
    :param exp: Explainer instance
    :param save_to: File name
    :param rename: True if input data must be renamed
    :param version: DiCE version
    :return:
    """

    colors = ['#1E88E5', '#FF0D57']
    fill_color = ['#c8e2f8', '#ffdae5']
    colors_attr = ['#779CB5'] * 4 + ['lightsteelblue'] * 8 + ['#E1EFF9'] * 6
    # Create strings of prediction and cf prediction
    prediction = 'Approved' if (exp.test_pred < 0.5) else 'Rejected'
    cf_prediction = 'Rejected' if (exp.test_pred < 0.5) else 'Approved'
    # Create df for original instance and cf explanations
    instance_df = reshape_instance(data, idx, rename)
    cf_df = reshape_dice_exp(exp, version, fill_empty)
    

    # List of values in required format
    value_list = [[value] for value in instance_df['Value'].to_list()]
    #value_list = [[value] for value in table_instance_df['Value'].to_list()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.6), gridspec_kw={'width_ratios': [1, 2.6]})

    # Table of original instance
    ax1.axis('tight')
    ax1.axis('off')
    table_ax1 = ax1.table(cellText=value_list,
                          colLabels=['Applicant Information'], colColours=['lightgrey'],
                          rowLabels=instance_df['Attribute'], rowColours=colors_attr,
                          loc="center",                          
                          cellLoc = 'center'
                          )  
    table_ax1.auto_set_font_size(False)
    table_ax1.set_fontsize(7)
    table_ax1.scale(1,1.1)
    for (row, col), cell in table_ax1.get_celld().items():
        if row == 0:
            cell.set_text_props(fontproperties=FontProperties(weight='bold', size=7))
    
    ax1_title = 'AI Recommendation       :' + '\n' + probability_to_text(prediction, exp.test_pred)
    fig.text(0.02, 0.9, ax1_title, ha='left', va='bottom', fontsize=11, color='black', fontweight='bold')
    fig.text(ax1.get_position().get_points()[1][0] + 0.03, 0.934, prediction, ha='right', va='bottom',
             fontsize=11, color=colors[int(exp.test_pred >= 0.5)], fontweight='bold')
    ax1.set_title('\n ', loc='left', fontweight='bold', pad=14)

    # Table of cf explanations
    colLabels = ['Attribute Changes ' + str(i + 1) for i in range(0, cf_df.shape[1])]
    

    ax2.axis('tight')
    ax2.axis('off')
    table_ax2 = ax2.table(cellText=cf_df.values, loc="center",
                          colLabels=colLabels,
                          colColours=['lightgrey'] * cf_df.shape[1],
                          cellLoc=alignment)
    table_ax2.auto_set_font_size(False)
    table_ax2.set_fontsize(7)
    table_ax2.scale(1,1.1)
    if fill_cell:
        for (row, col), cell in table_ax2.get_celld().items():
            if cell.get_text().get_text() != fill_empty and row != 0:
                cell.set_facecolor(fill_color[int(exp.test_pred < 0.5)])
            elif row == 0:
                cell.set_text_props(fontproperties=FontProperties(weight='bold', size=7))

    ax2_title = 'If the following attributes change       :\nthen the recommendation would be'
    ax2.set_title('\n ', loc='left', fontweight='bold', pad=14)
    fig.text(ax2.get_position().get_points()[0][0] + 0.025, 0.9, ax2_title, ha='left', va='bottom', fontsize=11, color='black')
    
    
    fig.text(ax2.get_position().get_points()[1][0] - 0.07, 0.934, cf_prediction, fontsize=11, 
             ha='right', va='bottom', color=colors[int(exp.test_pred < 0.5)])

    fontP = FontProperties()
    fontP.set_size('xx-small')
    
    loan_details = mpatches.Patch(color='#779CB5', label='Loan Details')
    financial_status = mpatches.Patch(color='lightsteelblue', label='Financial Status')
    personal_information = mpatches.Patch(color='#E1EFF9', label='Personal Information')
    ax1.legend(handles=[loan_details, financial_status, personal_information], bbox_to_anchor=(-0.7,-0.06), loc='lower left', ncol=3, prop=fontP)

    plt.tight_layout(w_pad=1.5)
    fig.savefig(save_to, dpi=180)
    plt.show()






def reshape_anchor_exp(exp):
    conditions = [re.sub('(\.0)0*', '', cond) for cond in exp.names()]
    connectors = ['IF'] + ['AND' for cond in conditions[:-1]]
    anchor_df = pd.DataFrame({'Connector': connectors, 'Condition': conditions})

    return anchor_df


def anchor_plot(data, idx, exp, proba, save_to, rename=False):
    from matplotlib.font_manager import FontProperties

    colors = ['#1E88E5', '#FF0D57']
    colors_attr = ['#779CB5'] * 4 + ['lightsteelblue'] * 8 + ['#E1EFF9'] * 6

    # Retrieve prediction
    prediction = 'Approved' if exp.exp_map['prediction'] == 0 else 'Rejected'
    # Create df for original instance
    instance_df = reshape_instance(data, idx, rename)
    anchor_df = reshape_anchor_exp(exp)

    # Create list of df values
    value_list = [[value] for value in instance_df['Value'].to_list()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.6), gridspec_kw={'width_ratios': [1, 2.6]})

    ax1.axis('tight')
    ax1.axis('off')
    table_ax1 = ax1.table(cellText=value_list,
                          colLabels=['Applicant Information'], colColours=['lightgrey'],
                          rowLabels=instance_df['Attribute'], rowColours=colors_attr,
                          loc="center",
                          cellLoc = 'center'
                          )  
    table_ax1.auto_set_font_size(False)
    table_ax1.set_fontsize(7)
    table_ax1.scale(1,1.1)

    for (row, col), cell in table_ax1.get_celld().items():
        if (row == 0):
            cell.set_text_props(fontproperties=FontProperties(weight='bold', size=7))

    ax1_title = 'AI Recommendation       :' + '\n' + probability_to_text(prediction, proba)
    fig.text(0.02, 0.9, ax1_title, ha='left', va='bottom', fontsize=11, color='black', fontweight='bold')
    fig.text(ax1.get_position().get_points()[1][0] + 0.03, 0.934, prediction, ha='right', va='bottom',
             fontsize=11, color=colors[exp.exp_map['prediction']], fontweight='bold')
    ax1.set_title('\n ', loc='left', fontweight='bold', pad=14)

    ax2.axis('tight')
    ax2.axis('off')

    table_ax2 = ax2.table(cellText=anchor_df.values,
                          loc="upper center", cellLoc='left', colWidths=[0.05, 0.95],
                          edges='open')
    table_ax2.auto_set_font_size(False)
    # table_ax2.set_fontsize(5)
    for (row, col), cell in table_ax2.get_celld().items():
        if col == 0:
            cell.set_text_props(fontproperties=FontProperties(weight='bold', size=9))
            # cell.set_color("#FF86AB")
        elif col == 1:
            cell.set_text_props(fontproperties=FontProperties(size=9))
            # cell.set_color('#FF86AB')
        if row % 3 != 0:
            #cell.set_text_props(fontproperties=FontProperties(weight='bold', size=80))
            cell.set_height(0.08)
            

    ax2_title = 'If the following conditions are fulfilled   :\nthen the AI recommends'
    fig.text(ax2.get_position().get_points()[1][0] - 0.03, 0.934, prediction, ha='right', va='bottom', color=colors[exp.exp_map['prediction']],
             fontsize=11)
    ax2.set_title('\n ', loc='left', fontweight='bold', pad=14)
    fig.text(ax2.get_position().get_points()[0][0] + 0.025, 0.9, ax2_title, ha='left', va='bottom', fontsize=11, color='black')

    fontP = FontProperties()
    fontP.set_size('xx-small')
    
    loan_details = mpatches.Patch(color='#779CB5', label='Loan Details')
    financial_status = mpatches.Patch(color='lightsteelblue', label='Financial Status')
    personal_information = mpatches.Patch(color='#E1EFF9', label='Personal Information')
    ax1.legend(handles=[loan_details, financial_status, personal_information], bbox_to_anchor=(-0.7,-0.06), loc='lower left', ncol=3, prop=fontP)
    plt.tight_layout(w_pad=1.5)

    fig.savefig(save_to, dpi=180)
    plt.show()


def lime_plot(data, idx, exp, save_to, rename=False, percentage=True, num_features = 6):
    colors = ['#1E88E5', '#FF0D57']
    colors_attr = ['#779CB5'] * 4 + ['lightsteelblue'] * 8 + ['#E1EFF9'] * 6
    
    # Create df for original instance
    instance_df = reshape_instance(data, idx, rename)
    
    # Create list of df values
    value_list = [[value] for value in instance_df['Value'].to_list()]
    
    prediction = 'Approved' if (exp.predict_proba[1] < 0.5) else 'Rejected'

    # Number of features

    # Computation of weights in %
    l = exp.as_list()
    s = pd.Series(dtype='float')
    for e in l:
        s[e[0]] = e[1]

    # Clean index by replacing floats with int
    idx_cleaned = [re.sub('(\.0)0*', '', idx) for idx in s.index]
    s.index = idx_cleaned

    feats = pd.DataFrame(index=s.index)
    feats['effect'] = s
    feats['abs_effect'] = s.abs()
    feats.sort_values(by='abs_effect', ascending=True, inplace=True)
    feats['percentage'] = feats['abs_effect'] / feats['abs_effect'].sum() * 100
    feats['percentage_d'] = feats['percentage'] * feats['effect'].apply(lambda x: -1 if x < 0 else 1)
    feats = feats.iloc[-num_features:, :]

    feats['color'] = [colors[1] if x > 0 else colors[0] for x in feats['effect'].values]
    feats['pos'] = np.arange(feats.shape[0]) + .5

    x_l = [0, 0]
    y_l = [0, feats.shape[0]]

    # separate feature name and values for equal symbols
    # manual modification
    idx = pd.Series(feats.index)
    idx = idx.apply(lambda x: insert_space(x))
    feats.index = idx.values
    
    feats = pad_df_index(feats, padding=55)
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.6), gridspec_kw={'width_ratios': [1, 1.4]}, constrained_layout=True)

    ax1.axis('tight')
    ax1.axis('off')
    table_ax1 = ax1.table(cellText=value_list,
                          colLabels=['Applicant Information'], colColours=['lightgrey'],
                          rowLabels=instance_df['Attribute'], rowColours=colors_attr,
                          loc="center",
                          cellLoc = 'center'
                          )  
    table_ax1.auto_set_font_size(False)
    
        
    table_ax1.set_fontsize(7)
    table_ax1.scale(1,1.1)
    
    for (row, col), cell in table_ax1.get_celld().items():
        if (row == 0):
            cell.set_text_props(fontproperties=FontProperties(weight='bold', size=7))

    ax1_title = 'AI Recommendation       :' + '\n' + probability_to_text(prediction, exp.predict_proba[1])
    fig.text(0.01, 0.9, ax1_title, ha='left', va='bottom', fontsize=11, color='black', fontweight='bold')
    fig.text(ax1.get_position().get_points()[1][0] - 0.06, 0.934, prediction, ha='right', va='bottom',
             fontsize=11, color=colors[int(exp.predict_proba[1] >= 0.5)], fontweight='bold')
    
    ax1.set_title('\n ', loc='left', fontweight='bold', pad=14)

    # ax2.axis('tight')
    if percentage:
        x_lim = feats['percentage'].max() * 1.4
        ax2.barh(feats['pos'], feats['percentage_d'], align='center', color=feats['color'], height=0.4)
    else:
        x_lim = feats['abs_effect'].max() * 1.4
        ax2.barh(feats['pos'], feats['effect'], align='center', color=feats['color'], height=0.6)
    ax2.plot(x_l, y_l, color='black', linewidth=2)
    ax2.set_yticks(feats['pos'])
    ax2.set_yticklabels(feats.index)
    ax2.tick_params(axis='both', which='major', labelsize=7)

    ax2.set_xlim(-x_lim, x_lim)
    # ax.set_xlim(-0.6, 0.6)
    # ax.set_xlim(-1.0, 1.0)
    ys, yf = ax2.get_ylim()
    ax2.set_ylim(ys, yf + 0.5)
    # ax2.set_xlabel('Influence %', fontsize=7)
    ax2.set_xticks([])

    if percentage:
        for i, f in feats.iterrows():
            if f.effect > 0:
                t = 1 / feats.shape[0] + x_lim * 0.125
            else:
                t = -1 / feats.shape[0] - x_lim * 0.125
            ax2.text(f.percentage_d + t, f.pos - feats.shape[0] * 0.01, '%.2f%%' % f.percentage, color=f.color,
                     horizontalalignment='center', fontsize=6)

        ax2.text(-x_lim / 2, feats['pos'].values[-1] + 0.8, exp.class_names[0], color=colors[0], fontsize=10,
                 horizontalalignment='center')
        ax2.text(x_lim / 2, feats['pos'].values[-1] + 0.8, exp.class_names[1], color=colors[1], fontsize=10,
                 horizontalalignment='center')

    else:
        for i, f in feats.iterrows():
            if f.effect > 0:
                t = x_lim * 0.07  # 1 / feats.shape[0] - x_lim * 0.39
            else:
                t = -x_lim * 0.07  # -1 / feats.shape[0] + x_lim * 0.39
            ax2.text(f.effect + t, f.pos - feats.shape[0] * 0.01, '{:.0%}'.format(f.abs_effect), color=f.color,
                     horizontalalignment='center', fontsize=8)

        ax2.text(-x_lim / 2, feats['pos'].values[-1] + 0.8, exp.class_names[0], color=colors[0], fontsize=10,
                 horizontalalignment='center')
        # ax.text(-x_lim/2, feats['pos'].values[-1] + 0.8, "Rejection", color=colors[0], fontsize=14, horizontalalignment='center')
        ax2.text(x_lim / 2, feats['pos'].values[-1] + 0.8, exp.class_names[1], color=colors[1], fontsize=10,
                 horizontalalignment='center')

    title = 'The following attributes influenced the recommendation of the AI:'
    
    plt.gcf().text(ax1.get_position().get_points()[1][0] + .015, 0.94 , title, fontsize=11, horizontalalignment='left')
    
    fontP = FontProperties()
    fontP.set_size('xx-small')
    
    loan_details = mpatches.Patch(color='#779CB5', label='Loan Details')
    financial_status = mpatches.Patch(color='lightsteelblue', label='Financial Status')
    personal_information = mpatches.Patch(color='#E1EFF9', label='Personal Information')
    ax1.legend(handles=[loan_details, financial_status, personal_information], bbox_to_anchor=(-0.5,-0.06), loc='lower left', ncol=3, prop=fontP)
    


    plt.savefig(save_to, dpi=180)


def pad_df_index(data, padding=55):
    df = data.copy()
    idx = pd.Series(df.index).apply(lambda x: f"{x:>{padding}}")
    #idx = pd.Series(df.index).apply(lambda x: "{:>60}".format(x))
    #idx = pd.Series(df.index).apply(lambda x: "{:<60}".format(x))
    df.index = idx
    return df
    
    
    





def shap_plot(data, idx, explainer, shap_values, predict_fn, save_to,
              class_names=None, num_features=7, height_bar=0.6,
                       link='identity', rename=False, figsize=(8.5, 4.2)):
    # Prepare data
    if class_names is None:
        class_names = ['Approved', 'Rejected']
    if rename:
        data = data.rename(columns=explanation_utils.ExplanationUtils.rename_dict)

    instance = data.loc[idx]
    feats = pd.DataFrame({'Value': instance, 'effect': shap_values, 'plot_effect': np.nan})
    base_value = explainer.expected_value
    out_value = np.sum(shap_values) + base_value
    y_prob = predict_fn(data.loc[idx].values)
    proba = [y_prob, 1 - y_prob]
    y_pred = (y_prob >= 0.5).astype('int32')[0]

    # Restore int values
    feats['Value'] = [re.sub('(\.0)0*', '', str(val)) for val in feats['Value']]

    colors_attr = ['#779CB5'] * 4 + ['lightsteelblue'] * 8 + ['#E1EFF9'] * 6
    # Create df for original instance
    instance_df = reshape_instance(data, idx, False)
    # Create list of df values
    value_list = [[value] for value in instance_df['Value'].to_list()]
    prediction = class_names[y_pred]

    # if y_pred == 1:
    #    colors = ['#FF0D57', '#1E88E5']
    #    color_separators = ['#FFC3D5', '#D1E6FA']
    #    arrows_text = ['lower', 'higher']
    # else:
    colors = ['#1E88E5', '#FF0D57']
    color_separators = ['#D1E6FA', '#FFC3D5']
    arrows_text = ['lower', 'higher']
    # Format negative features
    neg_feats = feats[feats['effect'] < 0]
    neg_feats = neg_feats.sort_values(by='effect', ascending=True)

    # Format postive features
    pos_feats = feats[feats['effect'] > 0].copy()
    pos_feats = pos_feats.sort_values(by='effect', ascending=False)

    # Define link function
    if link == 'identity':
        convert_func = lambda x: x
    elif link == 'logit':
        convert_func = lambda x: 1 / (1 + np.exp(-x))
    else:
        assert False, "ERROR: Unrecognized link function: " + str(link)

    # Convert negative feature values to plot values
    neg_val = out_value
    for i, r in neg_feats.iterrows():
        val = float(r['effect'])
        neg_val = neg_val + np.abs(val)
        feats.loc[i, 'plot_effect'] = convert_func(neg_val)
        neg_feats.loc[i, 'plot_effect'] = convert_func(neg_val)
    if len(neg_feats) > 0:
        total_neg = neg_feats['plot_effect'].max() - neg_feats['plot_effect'].min()
    else:
        total_neg = 0

    # Convert positive feature values to plot values
    pos_val = out_value
    for i, r in pos_feats.iterrows():
        val = float(r['effect'])
        pos_val = pos_val - np.abs(val)
        feats.loc[i, 'plot_effect'] = convert_func(pos_val)
        pos_feats.loc[i, 'plot_effect'] = convert_func(pos_val)
    if len(pos_feats) > 0:
        total_pos = pos_feats['plot_effect'].max() - pos_feats['plot_effect'].min()
    else:
        total_pos = 0

    # Convert output value and base value
    out_value = convert_func(out_value)
    base_value = convert_func(base_value)
    offset_text = (np.abs(total_neg) + np.abs(total_pos)) * 0.1

    # obtain index of effects greater than 0 and sorted ascending
    index = feats['effect'][feats['effect'] != 0].apply(abs).sort_values().index

    plot_feats = feats.loc[index]

    # find minimum between feat_limit and features with effect larger than 0
    num_features = np.min([num_features, plot_feats.shape[0]])

    height = 0.8
    pos = np.arange(num_features + 1) * np.round(height) + np.round(height) / 2

    plot_feats['color'] = [colors[1] if x > 0 else colors[0] for x in plot_feats['effect']]
    plot_feats['color_separator'] = [color_separators[1] if x > 0 else color_separators[0] for x in
                                     plot_feats['effect']]
    plot_feats['pos'] = pos[-1]
    plot_feats['rectangle'] = np.nan
    plot_feats['separator'] = np.nan

    # Define plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.6), gridspec_kw={'width_ratios': [1, 1.1]})

    ax1.axis('tight')
    ax1.axis('off')
    table_ax1 = ax1.table(cellText=value_list,
                          colLabels=['Applicant Information'], colColours=['lightgrey'],
                          rowLabels=instance_df['Attribute'], rowColours=colors_attr,
                          loc="center",
                          cellLoc = 'center'
                          )  
    table_ax1.auto_set_font_size(False)
    table_ax1.set_fontsize(7)
    table_ax1.scale(1,1.1)

    for (row, col), cell in table_ax1.get_celld().items():
        if (row == 0):
            cell.set_text_props(fontproperties=FontProperties(weight='bold', size=7))

    ax1_title = 'AI Recommendation       :' + '\n' + probability_to_text(prediction, y_prob)
    fig.text(0.02, 0.9, ax1_title, ha='left', va='bottom', fontsize=11, color='black', fontweight='bold')
    fig.text(ax1.get_position().get_points()[1][0] - 0.18, 0.934, prediction, ha='left', va='bottom',
             fontsize=11, color=colors[int(y_prob >= 0.5)], fontweight='bold')

    ax1.set_title('\n ', loc='left', fontweight='bold', pad=14)

    x_l = [out_value, out_value]
    y_l = [0, len(pos) * np.round(height)]
    ax2.plot(x_l, y_l, color='black', linewidth=1.5)

    update_axis_limits(ax2, plot_feats, total_pos, total_neg, base_value, out_value)
    width_separators = (ax2.get_xlim()[1] - ax2.get_xlim()[0]) / 200

    # Create bar for negative shap values
    features = draw_bars(out_value, plot_feats[plot_feats['effect'] < 0].sort_values(by='effect', ascending=True),
                         width_separators, height_bar=height_bar)
    plot_feats.loc[features.index, ['rectangle', 'separator']] = features[['rectangle', 'separator']]

    for i in features['rectangle'].values:
        ax2.add_patch(i)

    for i in features['separator'].values:
        ax2.add_patch(i)

    # Create bar for positive shap values
    features = draw_bars(out_value, plot_feats[plot_feats['effect'] > 0].sort_values(by='effect', ascending=False),
                         width_separators, height_bar=height_bar)
    plot_feats.loc[features.index, ['rectangle', 'separator']] = features[['rectangle', 'separator']]

    for i in features['rectangle'].values:
        ax2.add_patch(i)

    for i in features['separator'].values:
        ax2.add_patch(i)

    # find minimum between feat_limit and features with effect larger than 0
    pf = plot_feats.iloc[-num_features:, :].copy()
    pf['pos'] = pos[:-1]

    # Create bar for negative shap values
    features = draw_bars(out_value, pf[pf['effect'] < 0].sort_values(by='effect', ascending=True),
                         width_separators, height_bar=height_bar)
    pf.loc[features.index, ['rectangle', 'separator']] = features[['rectangle', 'separator']]

    for i in features['rectangle'].values:
        ax2.add_patch(i)

    for i in features['separator'].values:
        ax2.add_patch(i)

    # Create bar for negative shap values
    features = draw_bars(out_value, pf[pf['effect'] > 0].sort_values(by='effect', ascending=False),
                         width_separators, height_bar=height_bar)
    pf.loc[features.index, ['rectangle', 'separator']] = features[['rectangle', 'separator']]

    for i in features['rectangle'].values:
        ax2.add_patch(i)

    for i in features['separator'].values:
        ax2.add_patch(i)
    
    pf.index = pf.index + ' = ' + pf.Value.astype('str')
    pf = pad_df_index(pf, padding=55)
    ax2.set_yticks(pf.pos.values)
    ax2.set_yticklabels(pf.index)
    ax2.tick_params(axis='y', which='major', labelsize=7)
    ax2.tick_params(axis='x', which='major', labeltop=False, labelsize=7, length=3)

    for i, f in pf.iterrows():
        x = np.mean([f.rectangle.get_xy()[0][0], f.rectangle.get_xy()[1][0]])
        y = f.pos - height_bar * 0.88
        val = '%.2f' % (f['effect'])

        ax2.text(x, y, val, color=f.color, horizontalalignment='center', fontsize=5)

    out_value_offset = 0
    base_value_offset = 0

    diff = out_value - base_value
    if diff < 0.17 and diff > 0:
        # out_value_offset = (0.035 - diff) / 2
        base_value_offset = -(0.15 - diff) / 2 - 0.13
    elif diff > -0.17 and diff < 0:
        # out_value_offset = -(0.035 + diff) / 2
        base_value_offset = (0.15 + diff) / 2 + 0.13

    draw_output_element('Rejection Pr.', out_value, ax2, offset_text, colors, arrows_text, out_value_offset)
    # higher lower legend
    # draw_higher_lower_element(out_value, offset_text)

    # Add label for base value
    draw_base_element(base_value, ax2, base_value_offset)

    xlabel = get_xlabel(y_pred)
    title = "The AI's probability of rejection:" + '\n\n'

    plt.gcf().text(ax1.get_position().get_points()[1][0] + .015, 0.87, title, fontsize=11, horizontalalignment='left')
    
    fontP = FontProperties()
    fontP.set_size('xx-small')
    
    loan_details = mpatches.Patch(color='#779CB5', label='Loan Details')
    financial_status = mpatches.Patch(color='lightsteelblue', label='Financial Status')
    personal_information = mpatches.Patch(color='#E1EFF9', label='Personal Information')
    ax1.legend(handles=[loan_details, financial_status, personal_information], bbox_to_anchor=(-0.5,-0.06), loc='lower left', ncol=3, prop=fontP)
    
    plt.tight_layout(w_pad=1.5)
    plt.savefig(save_to, dpi=180)





def update_axis_limits(ax, plot_feats, total_pos, total_neg, base_value, out_value):
    padding = np.max([np.abs(total_pos) * 0.2,
                      np.abs(total_neg) * 0.2])

    pos_features = plot_feats[plot_feats['effect'] > 0]
    if len(pos_features) > 0:
        min_x = min(pos_features['plot_effect'].min(), base_value) - padding
    else:
        min_x = base_value - padding
    neg_features = plot_feats[plot_feats['effect'] < 0]
    if len(neg_features) > 0:
        max_x = max(neg_features['plot_effect'].max(), base_value) + padding
    else:
        max_x = out_value + padding
    ax.set_xlim(min_x, max_x)

    plt.tick_params(top=True, bottom=True, left=True, right=False, labelleft=True,
                    labeltop=True, labelbottom=True)
    # plt.tick_params(top=True, bottom=False, left=False, right=False, labelleft=False,
    #                labeltop=True, labelbottom=False)
    # plt.locator_params(axis='x', nbins=12)

    # for key, spine in zip(plt.gca().spines.keys(), plt.gca().spines.values()):
    #    if key != 'top':
    #        spine.set_visible(False)

    return min_x, max_x


def draw_bars(out_value, features, width_separators, height_bar):
    """Draw the bars and separators."""
    pre_val = out_value
    for i in range(features.shape[0]):
        index = features.index[i]
        f = features.loc[index]
        if f.effect > 0:
            left_bound = f.plot_effect
            right_bound = pre_val
            pre_val = left_bound

            separator_indent = np.abs(width_separators)
            separator_pos = left_bound
            # colors = ['#FF0D57', '#FFC3D5']
        else:
            left_bound = pre_val
            right_bound = f.plot_effect
            pre_val = right_bound

            separator_indent = - np.abs(width_separators)
            separator_pos = right_bound
            # colors = ['#1E88E5', '#D1E6FA']

        # Create rectangle
        if i == 0:
            if f.effect > 0:
                points_rectangle = [[left_bound, f.pos - height_bar / 2],
                                    [right_bound, f.pos - height_bar / 2],
                                    [right_bound, f.pos + height_bar / 2],
                                    [left_bound, f.pos + height_bar / 2],
                                    [left_bound + separator_indent, f.pos]]
            else:
                points_rectangle = [[right_bound, f.pos - height_bar / 2],
                                    [left_bound, f.pos - height_bar / 2],
                                    [left_bound, f.pos + height_bar / 2],
                                    [right_bound, f.pos + height_bar / 2],
                                    [right_bound + separator_indent, f.pos]]

        else:
            points_rectangle = [[left_bound, f.pos - height_bar / 2],
                                [right_bound, f.pos - height_bar / 2],
                                [right_bound + separator_indent * 0.90, f.pos],
                                [right_bound, f.pos + height_bar / 2],
                                [left_bound, f.pos + height_bar / 2],
                                [left_bound + separator_indent * 0.90, f.pos]]

        line = plt.Polygon(points_rectangle, closed=True, fill=True,
                           facecolor=f.color, linewidth=0)
        features.loc[index, 'rectangle'] = line

        # Create seperator
        points_separator = [[separator_pos, f.pos - height_bar / 2],
                            [separator_pos + separator_indent, f.pos],
                            [separator_pos, f.pos + height_bar / 2]]

        line = plt.Polygon(points_separator, closed=None, fill=None,
                           edgecolor=f.color_separator, lw=0.8)
        features.loc[index, 'separator'] = [line]

    return features


def draw_output_element(out_name, out_value, ax, offset_text, colors, arrows_text, out_value_offset=0):
    # Add output value
    ys, yf = ax.get_ylim()
    r = yf - ys
    x, y = np.array([[out_value, out_value], [yf + 0.01, yf + r / 40]])
    line = lines.Line2D(x, y, lw=1.5, color='black')
    line.set_clip_on(False)
    ax.add_line(line)

    font0 = FontProperties()
    font = font0.copy()
    font.set_weight('bold')
    # text_out_val = plt.text(out_value - 0.013, yf + r/30 + 0.55, '{0:.2f}'.format(out_value),
    # text_out_val = plt.text(out_value, yf + r/30 + 0.55, '{0:.2f}'.format(out_value),
    text_out_val = plt.text(out_value + out_value_offset, yf + r / 30 + 0.1, '{0:.2f}'.format(out_value),
                            fontproperties=font,
                            fontsize=7,
                            horizontalalignment='center')
    # text_out_val.set_bbox(dict(facecolor='white', edgecolor='white'))

    # text_out_val = plt.text(out_value - 0.013, yf + r/30 + 0.9, out_name,
    # text_out_val = plt.text(out_value, yf + r/30 + 0.9, out_name,
    text_out_val = plt.text(out_value + out_value_offset, yf + r / 30 + 0.45, out_name,
                            fontsize=7, alpha=1.0,
                            horizontalalignment='center')
    # text_out_val.set_bbox(dict(facecolor='white', edgecolor='white'))

    ax.text(out_value - offset_text, yf + r / 30 + 0.85, arrows_text[1],
            fontsize=7, color=colors[1],
            horizontalalignment='right')

    ax.text(out_value + offset_text, yf + r / 30 + 0.85, arrows_text[0],
            fontsize=7, color=colors[0],
            horizontalalignment='left')

    ax.text(out_value, yf + r / 30 + 0.78, r'$\leftarrow$',
            fontsize=7, color=colors[0],
            horizontalalignment='center')

    ax.text(out_value, yf + r / 30 + 0.92, r'$\rightarrow$',
            fontsize=7, color=colors[1],
            horizontalalignment='center')


def draw_base_element(base_value, ax, base_value_offset=0):
    ys, yf = ax.get_ylim()
    r = yf - ys

    if base_value_offset == 0:
        x, y = np.array([[base_value, base_value], [yf + 0.01, yf + r / 40]])
        # line = lines.Line2D(x, y, lw=2., color='#F2F2F2')
        line = lines.Line2D(x, y, lw=1.5, color='black')
        line.set_clip_on(False)
        ax.add_line(line)

    if base_value_offset != 0:
        x, y = np.array([[base_value, base_value], [yf + 0.01, yf + r / 50]])
        # line = lines.Line2D(x, y, lw=2., color='#F2F2F2')
        line = lines.Line2D(x, y, lw=1.5, color='black')
        line.set_clip_on(False)
        ax.add_line(line)

        x, y = np.array([[base_value, base_value + base_value_offset], [yf + r / 50, yf + r / 50]])
        line = lines.Line2D(x, y, lw=1.5, color='black')
        line.set_clip_on(False)
        ax.add_line(line)

        x, y = np.array([[base_value + base_value_offset, base_value + base_value_offset], [yf + r / 50, yf + r / 40]])
        # line = lines.Line2D(x, y, lw=2., color='#F2F2F2')
        line = lines.Line2D(x, y, lw=1.5, color='black')
        line.set_clip_on(False)
        ax.add_line(line)

    font0 = FontProperties()
    font = font0.copy()
    font.set_weight('bold')
    # text_base_val = plt.text(base_value + 0.013, yf + r/30 + 0.55, '{0:.2f}'.format(base_value),
    # text_base_val = plt.text(base_value, yf + r/30 + 0.55, '{0:.2f}'.format(base_value),
    text_base_val = plt.text(base_value + base_value_offset, yf + r / 30 + 0.1, '{0:.2f}'.format(base_value),
                             fontproperties=font,
                             fontsize=7,
                             horizontalalignment='center')
    # text_base_val.set_bbox(dict(facecolor='white', edgecolor='white'))

    # text_base_val = plt.text(base_value + 0.013, yf + r/30 + 0.9, 'Base Pr.',
    # text_base_val = plt.text(base_value, yf + r/30 + 0.9, 'Base Pr.',
    text_base_val = plt.text(base_value + base_value_offset, yf + r / 30 + 0.45, 'Base Pr.',
                             fontsize=7, alpha=1.0,
                             horizontalalignment='center')
    # text_base_val.set_bbox(dict(facecolor='white', edgecolor='white'))


def get_xlabel(y_pred):
    if y_pred == 1:
        xlabel = 'Approval'
    else:
        xlabel = 'Rejection'

    return xlabel
