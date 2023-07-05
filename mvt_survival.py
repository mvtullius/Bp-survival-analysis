'''MIT License

Copyright (c) 2023 Michael Tullius

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.'''


from itertools import combinations
from collections import namedtuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from scipy.stats import barnard_exact
from lifelines import KaplanMeierFitter
import PyPDF2
from IPython.display import display, HTML


class SaveDataFrames:
    '''A simple class for storing Pandas DataFrames in a Dictionary and writing them to separate sheets in an Excel file.'''

    def __init__(self, output_excel_filename):
        self.data_frames = {}
        self.filename = output_excel_filename

    
    def add_data_frame(self, df, name):
        self.data_frames[name] = df.copy()

    def write(self):

        writer = pd.ExcelWriter(self.filename)
    
        for key, df in self.data_frames.items():
            print(f'Saving DataFrame: {key}')
            df.to_excel(writer, sheet_name=key)
        
        writer.close()
        

def display_df(input_df, name):
    '''Display a Pandas DataFrame in HTML format with a name in large font and with spacing.'''
    
    display(HTML(f'<h1>{name}</h1><br>'))
    display(input_df)
    display(HTML('<br>'))


def concat_pdf_files(pdf_file_list, output_file_name):
    '''Adapted from Automate the Boring Stuff book.  Combine multiple pdf files into a single file.'''
    
    pdf_writer = PyPDF2.PdfFileWriter()
    
    pdf_files = (open(pdf, 'rb') for pdf in pdf_file_list)
    
    for pdf_file in pdf_files:
        
        pdf_reader = PyPDF2.PdfFileReader(pdf_file)
        
        for page_num in range(pdf_reader.numPages):
            page_obj = pdf_reader.getPage(page_num)
            pdf_writer.addPage(page_obj)
            
    new_pdf = open(output_file_name, 'wb')
    pdf_writer.write(new_pdf)
    new_pdf.close()


class AlphaColor:
    '''AlphaColor is used to set a color based on the p-value in relation to different alpha values.
    Three dictionaries are passed to the constructor.  The keys are the p-value levels (e.g. 0.05, 0.01, 0.001, etc)
    and the values are the colors (a tuple with a Matplotlib color name and an alpha).'''
    
    def __init__(self, most_stringent_correction_p_value_alphas, less_stringent_correction_p_value_alphas, uncorrected_p_value_alphas, default_color = 'whitesmoke'):
                   
        self.most_stringent_correction_p_value_alphas = most_stringent_correction_p_value_alphas
        self.less_stringent_correction_p_value_alphas = less_stringent_correction_p_value_alphas
        self.uncorrected_p_value_alphas = uncorrected_p_value_alphas

        self.most_stringent_correction_p_value_alphas_sorted = sorted(self.most_stringent_correction_p_value_alphas.keys())
        self.less_stringent_correction_p_value_alphas_sorted = sorted(self.less_stringent_correction_p_value_alphas.keys())
        self.uncorrected_p_value_alphas_sorted = sorted(self.uncorrected_p_value_alphas.keys())

        self.default_color = default_color
            
            
    def get_color(self, most_stringent_correction_p_value, less_stringent_correction_p_value, uncorrected_p_value):

        # first check corrected p-value levels and then check uncorrected p-values if corrected p-value is not less than any of the thresholds
             
        # holm = most stringent correction
        
        for alpha_level_key in self.most_stringent_correction_p_value_alphas_sorted:
            color = self.most_stringent_correction_p_value_alphas[alpha_level_key]
            if most_stringent_correction_p_value < alpha_level_key:
                return color
            
        # fdr = less stringent correction
        
        for alpha_level_key in self.less_stringent_correction_p_value_alphas_sorted:
            color = self.less_stringent_correction_p_value_alphas[alpha_level_key]
            if less_stringent_correction_p_value < alpha_level_key:
                return color
          
        # no correction
        
        for alpha_level_key in self.uncorrected_p_value_alphas_sorted:
            color = self.uncorrected_p_value_alphas[alpha_level_key]
            if uncorrected_p_value < alpha_level_key:
                return color
            
        return (self.default_color, 1)
    

def create_p_value_legend_figure(correction_methods, p_value_colors, filename):
    '''Create an image that can be used as a figure legend for plots that are color coded by the significance level of the p-value.
    There should be two correction methods, the first one more stringent than the second. p_value_colors is an AlphaColor object.'''
    
    most_stringent_correction = correction_methods[0]
    less_stringent_correction = correction_methods[1]
    
    most_stringent_correction_colors = p_value_colors.most_stringent_correction_p_value_alphas
    less_stringent_correction_colors = p_value_colors.less_stringent_correction_p_value_alphas
    uncorrected_p_value_colors = p_value_colors.uncorrected_p_value_alphas
    
    
    fig, ax1 = plt.subplots(1, figsize=(5,5))

    y_values = [300-i*20 for i in range(len(most_stringent_correction_colors))]

    for (p_value, (color, alpha)), y_value in zip(most_stringent_correction_colors.items(), y_values):

        annotation = ax1.annotate(f'{most_stringent_correction} adjusted p-value < {p_value}', xy=(0, 0), xycoords='figure fraction',
                     xytext=(50, y_value), textcoords='offset points',
                     ha="left", va="bottom",
                     bbox=dict(boxstyle="round"))

        box = annotation.get_bbox_patch()
        box.set_color(color)
        box.set_alpha(alpha)
        box.set_edgecolor('black')

    y_values = [200-i*20 for i in range(len(less_stringent_correction_colors))]

    for (p_value, (color, alpha)), y_value in zip(less_stringent_correction_colors.items(), y_values):

        annotation = ax1.annotate(f'{less_stringent_correction} adjusted p-value < {p_value}', xy=(0, 0), xycoords='figure fraction',
                     xytext=(50, y_value), textcoords='offset points',
                     ha="left", va="bottom",
                     bbox=dict(boxstyle="round"))

        box = annotation.get_bbox_patch()
        box.set_color(color)
        box.set_alpha(alpha)
        box.set_edgecolor('black')

    y_values = [100-i*20 for i in range(len(uncorrected_p_value_colors))]

    for (p_value, (color, alpha)), y_value in zip(uncorrected_p_value_colors.items(), y_values):

        annotation = ax1.annotate(f'Uncorrected p-value < {p_value}', xy=(0, 0), xycoords='figure fraction',
                     xytext=(50, y_value), textcoords='offset points',
                     ha="left", va="bottom",
                     bbox=dict(boxstyle="round"))

        box = annotation.get_bbox_patch()
        box.set_color(color)
        box.set_alpha(alpha)
        box.set_edgecolor('black')

        ax1.set_axis_off()
        
    fig.tight_layout()
    fig.savefig(filename, bbox_inches='tight')
    
    return fig
    

def get_corrected_p_values(uncorrected_p_values, methods=None, alpha=0.05):
    '''Uses statsmodels.stats.multitest.multipletests to calculate corrected p-values.  If methods is None (default), all available methods are used.
    Returns a Pandas DataFrame with the results.'''
    
    if methods == None:
        # All available methods for statsmodels.stats.multitest.multipletests
        # https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
        
        methods = ['bonferroni', # : one-step correction
                       'sidak',      # : one-step correction
                       'holm-sidak', # : step down method using Sidak adjustments
                       'holm', # : step-down method using Bonferroni adjustments
                       'simes-hochberg', # : step-up method (independent)
                       'hommel', # : closed method based on Simes tests (non-negative)
                       'fdr_bh', # : Benjamini/Hochberg (non-negative)
                       'fdr_by', # : Benjamini/Yekutieli (negative)
                       'fdr_tsbh', # : two stage fdr correction (non-negative)
                       'fdr_tsbky'] # : two stage fdr correction (non-negative)
        
    
    corrected_p_values = {'total_p_values': len(uncorrected_p_values),
                          'significant_without_correction': sum(uncorrected_p_values<alpha)}

    for method in methods:
        reject_null_hypothesis, pvals_corrected, alpha_corrected_Sidak, alpha_corrected_Bonf = multipletests(uncorrected_p_values.sort_values(),
                                                                                                             alpha=alpha,
                                                                                                             is_sorted=True,
                                                                                                             method=method)
        
        corrected_p_values['significant_with_correction_' + method] = sum(reject_null_hypothesis)
        corrected_p_values['reject_null_hypothesis_' + method] = reject_null_hypothesis
        corrected_p_values['p_value_with_correction_' + method] = pvals_corrected
        corrected_p_values['alpha_corrected_Sidak_' + method] = alpha_corrected_Sidak
        corrected_p_values['alpha_corrected_Bonf_' + method] = alpha_corrected_Bonf
        
    return pd.DataFrame(corrected_p_values)
    

def create_stats_df(df, uncorrected_p_values):
    '''Create a dataframe with uncorrected p-values generated from a Kaplan Meier pairwise_logrank_test (lifelines package).'''
    
    stats_df = (
        pd.DataFrame(combinations(df.Group.unique(), 2), columns=['Group_1', 'Group_2'])
            .assign(pair=lambda df_: df_.Group_1 + "-" + df_.Group_2)
            .assign(uncorrected_p_value=uncorrected_p_values)
    )
    return stats_df


def create_stats_df_with_corrected_p_values(df, rows_to_drop=[]):
    '''Calculate p-values corrected for multiple comparisons. Input df is from create_stats_df function.
      
      Set rows_to_drop parameter to an index if only a subset of comparisons are desired.'''
    
    # data is sorted by p_value
    # need to insert the corrected p-values (also in sorted order) without regard to the original index, thus the need to 
    #   first .reset_index() and then at the end .set_index()/.sort_index()
    
    corrected_p_values_df = get_corrected_p_values(df.uncorrected_p_value)
    
    stats_df = (
            df
            .drop(rows_to_drop)
            .sort_values('uncorrected_p_value')
            .reset_index()
            .assign(p_value_fdr_bh=corrected_p_values_df.p_value_with_correction_fdr_bh)
            .assign(p_value_holm_bonferroni=corrected_p_values_df['p_value_with_correction_holm'])
            .assign(p_value_holm_sidak=corrected_p_values_df['p_value_with_correction_holm-sidak'])
            .set_index('index')
            .sort_index()
            .assign(sig_uncorrected=lambda df_: pd.cut(df_['uncorrected_p_value'],
                               bins=[0, 0.0001, 0.001, 0.01, 0.05, 1],
                               labels=['****', '***', '**', '*', 'ns']))
            .assign(sig_fdr_bh=lambda df_: pd.cut(df_['p_value_fdr_bh'],
                               bins=[0, 0.0001, 0.001, 0.01, 0.05, 1],
                               labels=['****', '***', '**', '*', 'ns']))
            .assign(sig_holm_bonferroni=lambda df_: pd.cut(df_['p_value_holm_bonferroni'],
                                bins=[0, 0.0001, 0.001, 0.01, 0.05, 1],
                                labels=['****', '***', '**', '*', 'ns']))
            .assign(sig_holm_sidak=lambda df_: pd.cut(df_['p_value_holm_sidak'],
                                bins=[0, 0.0001, 0.001, 0.01, 0.05, 1],
                                labels=['****', '***', '**', '*', 'ns']))
    )
    return stats_df  


def kmf_fit_group(df_, group):
    '''kmf_fit_group performs a Kaplan Meier fit (lifelines package) for a selected group.'''
    
    return KaplanMeierFitter().fit(df_.loc[df_.Group == group, "Day"],
                                   df_.loc[df_.Group == group, "Death"], label=group)


def get_stats_from_kmf(group, kmf_fit):
    '''get_stats_from_kmf returns a Pandas Dataframe with nicely formatted statistics from a Kaplan Meier fit (lifelines package) for a selected group.'''
    
    columns = ['final_survival', 'upper_0.95', 'lower_0.95', 'median_survival_time']
    
    return (
        kmf_fit.confidence_interval_
        .iloc[-1]
        .to_frame()
        .T
        .set_axis(['lower_0.95', 'upper_0.95'], axis='columns')
        .assign(final_survival=kmf_fit.survival_function_.iloc[-1].values[0])
        .assign(median_survival_time=kmf_fit.median_survival_time_)
        .assign(Group=group)
        .set_index('Group')
    )[columns]


def create_kmf_survival_figure(df, ax, groups=None):
    '''Return a matplotlib figure using kmf.plot_survival_function.
    Plot all groups in df by deault. Set groups parameter to a list of group names to plot selected groups.'''
    
    if groups:
        mask = df.Group.isin(groups)
        df_select = df[mask]
    else:
        df_select = df

    for name, grouped_df in df_select.groupby('Group'):
        kmf = KaplanMeierFitter()
        kmf.fit(grouped_df["Day"], grouped_df["Death"], label=name)
        kmf.plot_survival_function(ax=ax, ci_alpha=0.2)

    
def create_pairwise_plot(df, combined_stats_df, alpha_color, pairwise_comparisons, layout, p_value_image=None):
    
    columns = len(layout[0])
    rows = len(layout)
    
    fig = plt.figure(constrained_layout=True, figsize=(columns*3, rows*3))
    ax_dict = fig.subplot_mosaic(layout)

    for pair in pairwise_comparisons:
        group_1 = pair.Group_1
        group_2 = pair.Group_2
        key = f'{group_1}-{group_2}'
        ax = ax_dict[key]

        mask = (df.Group == group_1) | (df.Group == group_2)
        df_select = df[mask]

        for name, grouped_df in df_select.groupby('Group'):
            kmf = KaplanMeierFitter()
            kmf.fit(grouped_df["Day"], grouped_df["Death"], label=name)
            
            kmf.plot_survival_function(ax=ax, ci_alpha=0.2)
            ax.set_ylim(0, 1.05)

            mask2 = (combined_stats_df.pair == key)
            p_value = combined_stats_df.loc[mask2, 'uncorrected_p_value'].values[0]
            holm_adj_p_value = combined_stats_df.loc[mask2, 'p_value_holm_bonferroni'].values[0]
            fdr_adj_p_value = combined_stats_df.loc[mask2, 'p_value_fdr_bh'].values[0]

            color, color_alpha = alpha_color.get_color(holm_adj_p_value, fdr_adj_p_value, p_value)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.7), ncol=2,
                                 facecolor=color, framealpha=color_alpha)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            annotation = ax.annotate(f'p = {p_value:0.4f}\nfdr_adj_p = {fdr_adj_p_value:0.4f}\nholm_bonferroni_adj_p = {holm_adj_p_value:0.4f}',
                xy=(0, 0), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"),
                xytext=(0.8, 1.36), textcoords='axes fraction',
                horizontalalignment='right', verticalalignment='top')

            box = annotation.get_bbox_patch()
            box.set_color(color)
            box.set_alpha(color_alpha)
            box.set_edgecolor('black')
            
            
    top_row = layout[0]
    last_column = [row[-1] for row in layout]

    for column_name in top_row:
        ax_dict[column_name].set_axis_off()
        if column_name != 'top_right':
            label = column_name.split('_')[1]
            annotation = ax_dict[column_name].annotate(f'{label}',
                    xy=(0, 0), xycoords='axes fraction',
                    xytext=(0.5, 0.5), textcoords='axes fraction',
                    horizontalalignment='center', verticalalignment='top', fontsize=16)
    
    for row_name in last_column:
        ax_dict[row_name].set_axis_off()
        if row_name != 'top_right':
            label = row_name.split('_')[1]
            annotation = ax_dict[row_name].annotate(f'{label}',
                xy=(0, 0), xycoords='axes fraction',
                xytext=(0.5, 0.5), textcoords='axes fraction',
                horizontalalignment='center', verticalalignment='top', fontsize=16)
            
    for row in layout:
        pairwise_comparisons_strings = [f'{p.Group_1}-{p.Group_2}' for p in pairwise_comparisons]
        pairs = [pair for pair in row if pair in pairwise_comparisons_strings]
        for pair in pairs[1:]:
            # remove xlabel for all but the first axis for each row
            ax_dict[pair].set_xlabel('')
          
    if p_value_image is not None:
        # Place image of p-value levels on figure
        width = fig.bbox.width
        height = fig.bbox.height

        x = width/2 - 2*p_value_image.shape[0]
        y = 0.4*height - p_value_image.shape[1]

        fig.figimage(p_value_image, x, y, zorder=1, origin='upper')
      
    return fig, ax_dict


def get_pairwise_comparisons_and_mosaic_layout(groups):
    '''Return all pairwise comparisons for a list of groups, along with a layout list-of-lists for matplotlib fig.subplot_mosaic()'''
    
    Pair = namedtuple("Pair", ['Group_1', 'Group_2'])
    
    pairwise_comparisons = [Pair(gr1, gr2) for (gr1, gr2) in combinations(groups, 2)]
    
    pairwise_comparisons_layout = []
    
    for i, group in enumerate(groups[:-1]):
        row_layout = i * ['.'] + [f'{gr1}-{gr2}' for (gr1, gr2) in combinations(groups[i:], 2) if gr1 == groups[i]]
        pairwise_comparisons_layout.append(row_layout)
        
    top_row = [f'column_{group}' for group in groups[1:]] + ['top_right']
    last_column = [[f'row_{group}'] for group in groups]  
    
    mosaic_layout = []
    for row, lc in zip(pairwise_comparisons_layout, last_column):
        mosaic_layout.append(row+lc)
    mosaic_layout.insert(0, top_row)

    return pairwise_comparisons, mosaic_layout



def create_proportion_and_stats_df(raw_data_df, pairwise_comparisons='all'):
    '''Calculate percent survival and percent sterile immunity from the the input dataframe.
    Perform Barnard exact tests using scipy.stats.barnard_exact.'''
    
    results = dict()
    
    died = (
        raw_data_df
        .groupby('Group')
        .Death
        .sum()
        .rename('Died')
    )

    total_mice = (
        raw_data_df
        .groupby('Group')
        .Death
        .count()
        .rename('total_mice')
    )

    survived = total_mice - died
    survived.name = 'Survived'

    sterile = (
        raw_data_df
        .groupby('Group')
        .Survival_with_sterile_immunity
        .sum()
    )

    sterile_immunity_df = pd.concat([died, survived, total_mice, sterile], axis=1, verify_integrity=True).reset_index()
  
    proportion_df = (
        sterile_immunity_df
        .assign(not_sterile = lambda df_: df_.total_mice - df_.Survival_with_sterile_immunity)
        .assign(percent_survival = lambda df_: 100*df_.Survived / df_.total_mice)
        .assign(percent_sterile = lambda df_: 100*df_.Survival_with_sterile_immunity / df_.total_mice)
    )
    
    results['Proportion'] = proportion_df
    
    if pairwise_comparisons == 'all':
        groups = proportion_df.Group.unique()
        pairwise_comparisons = [(gr1, gr2) for (gr1, gr2) in combinations(groups, 2)]
        

    for col_name in ['Survived', 'Survival_with_sterile_immunity']:
        stats_df = pd.DataFrame([f'{gr1}-{gr2}' for (gr1, gr2) in pairwise_comparisons], columns=['Pair'])
        stats_df['uncorrected_p_value_barnard'] = 1.0

        for (gr1, gr2) in pairwise_comparisons:
            key = f'{gr1}-{gr2}'
            
            if col_name == 'Survived':
                remaining_mice = 'Died'
            elif col_name == 'Survival_with_sterile_immunity':
                remaining_mice = 'not_sterile'
                
            contingency_table = proportion_df.set_index('Group').loc[[gr1, gr2], [remaining_mice, col_name]]
            
            barnard_results = barnard_exact(contingency_table, alternative='two-sided', pooled=True, n=32)
            stats_df.loc[stats_df.Pair == key, 'uncorrected_p_value_barnard'] = barnard_results.pvalue

        
        corrected_p_values_df = get_corrected_p_values(stats_df.uncorrected_p_value_barnard)
        
        results[col_name] = (
            stats_df
            .sort_values('uncorrected_p_value_barnard')
            .reset_index()
            .assign(p_value_barnard_fdr_bh=corrected_p_values_df.p_value_with_correction_fdr_bh)
            .assign(p_value_barnard_holm_bonferroni=corrected_p_values_df['p_value_with_correction_holm'])
            .assign(p_value_barnard_holm_sidak=corrected_p_values_df['p_value_with_correction_holm-sidak'])
            .fillna(1.0)
            .set_index('index')
            .sort_index()
            .assign(sig_barnard_uncorrected=lambda df_: pd.cut(df_['uncorrected_p_value_barnard'],
                               bins=[0, 0.0001, 0.001, 0.01, 0.05, 1],
                               labels=['****', '***', '**', '*', 'ns']))
            .assign(sig_barnard_fdr_bh=lambda df_: pd.cut(df_['p_value_barnard_fdr_bh'],
                               bins=[0, 0.0001, 0.001, 0.01, 0.05, 1],
                               labels=['****', '***', '**', '*', 'ns']))
            .assign(sig_barnard_holm_bonferroni=lambda df_: pd.cut(df_['p_value_barnard_holm_bonferroni'],
                                bins=[0, 0.0001, 0.001, 0.01, 0.05, 1],
                                labels=['****', '***', '**', '*', 'ns']))
            .assign(sig_barnard_holm_sidak=lambda df_: pd.cut(df_['p_value_barnard_holm_sidak'],
                                bins=[0, 0.0001, 0.001, 0.01, 0.05, 1],
                                labels=['****', '***', '**', '*', 'ns']))
            .reindex(columns=['Pair',
                              'uncorrected_p_value_barnard', 'p_value_barnard_fdr_bh',
                              'p_value_barnard_holm_bonferroni', 'p_value_barnard_holm_sidak',
                              'sig_barnard_uncorrected', 'sig_barnard_fdr_bh',
                              'sig_barnard_holm_bonferroni', 'sig_barnard_holm_sidak'])
        )
    
    return results

        
def main():
    print('''mvt_survival.py:
    
    This file is not meant to be executed directly.  It contains helper functions for analyzing survival data.''')

if __name__ == '__main__':
    main()
