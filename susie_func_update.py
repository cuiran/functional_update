import os
import pandas as pd
import pickle
import glob
import warnings
import gzip
import numpy as np
from scipy import stats
warnings.filterwarnings('ignore')
#load SuSiE R package (otherwise we get a segmentation fault when we open the pickle files sometimes...)
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects as ro
ro.conversion.py2ri = numpy2ri
numpy2ri.activate()
from rpy2.robjects.packages import importr
susieR = importr('susieR')
import argparse


def load_snpvar(trait, chr_num, num_bins=20, min_var_ratio=100):
    '''
    Return a dataframe for the per-snp variance of the given chromosome

    Columns: ['SNPVAR','v'], where v is the variant ID chr:pos:a1:a2
    '''
    coef_dir = os.path.join(COEF_DIR_ROOT, 'CKmedian_%dbins'%(num_bins))
    bins_annot_file = os.path.join(ANNOT_DIR_ROOT, '%s.b%d.%d.annot.parquet'%(trait, num_bins, chr_num))
    ldsc_coef_file = os.path.join(COEF_DIR_ROOT, '%s.337K.evenodd.gz'%(trait))

    #compute snpvar
    df_annotations = pd.read_parquet(bins_annot_file)
    num_funct_bins = df_annotations.shape[1]
    df_ldsc_taus = pd.read_table(ldsc_coef_file, index_col=['Chromosome'])
    df_ldsc_taus = df_ldsc_taus.loc[chr_num]
    df_ldsc_taus.index = df_ldsc_taus.index.str[:-2]
    assert (df_annotations.columns == df_ldsc_taus.index).all()
    df_snpvar = df_annotations.dot(df_ldsc_taus)
    h2_total = df_snpvar.sum()
    df_snpvar[df_snpvar <= df_snpvar.max() / min_var_ratio] = df_snpvar.max() / min_var_ratio
    df_snpvar *= (h2_total / df_snpvar.sum())
    
    return df_snpvar


def load_results_d(trait,region,result_type="unif"):
    #set file name according to result_type
    if result_type=="unif":
        result_file = os.path.join(RESULT_DIR_ROOT,
                        '%s/%s/susie_L10_priovar-2.000_noninf/susie.pickle.gz'%(trait,region))
    elif result_type=="func":
        result_file = os.path.join(RESULT_DIR_ROOT,
                        '%s/%s/susie_L10_priovar-2.000_func_20bins_minratio100_noninf/susie.pickle.gz'%(trait,region))
    #read in pickle.gz file as a dictionary
    if os.path.exists(result_file):
        with gzip.GzipFile(result_file,'rb') as f:
            result_d = pickle.load(f)
    else:
        warnings.warn('No {} fine-mapping results for region {} trait {}'.format(result_type,region,trait))
        result_d = dict()
    
    return result_d


def func_update_one_causal(unif_pip, priors):
    func_pip_unscaled = np.multiply(unif_pip.flatten(),priors.flatten())
    func_pip = func_pip_unscaled/np.sum(func_pip_unscaled)

    return func_pip


def save_array(a,col_name="default_colname",file_name="default_fname"):
    df = pd.DataFrame(a,columns=[col_name])
    df.to_csv(file_name,sep='\t',index=False)
    return

def save_matrix(a,col_name="",file_name="default_fname"):
    df = pd.DataFrame(a,columns=col_name)
    df.to_csv(file_name,sep='\t',index=False)
    return

def compute_func_update(trait):
    regions = [os.path.basename(f) for f in glob.glob(RESULT_DIR_ROOT+trait+'/chr*')]
    print('There are {} regions in {} folder'.format(len(regions),trait))
    for region in regions[:100]:
        alpha_fname = RESULT_DIR_ROOT+trait+'/'+region+'/'+'susie_L10_priovar-2.000_func_20bins_minratio100_noninf/func_update_alpha'
        pip_fname = RESULT_DIR_ROOT+trait+'/'+region+'/'+'susie_L10_priovar-2.000_func_20bins_minratio100_noninf/func_update_pip'
        if os.path.exists(alpha_fname) and os.path.exists(pip_fname):
            continue
            
        unif_d = load_results_d(trait,region)
        if not unif_d:
            continue
        func_d = load_results_d(trait,region,result_type='func')
        clear_output(wait=True)
        display('Computing analytical update for region {}, {} out of {}'.format(region,regions.index(region),len(regions)))
        func_update_alpha = np.array([func_update_one_causal(alpha,func_d['pi']) for alpha in unif_d['alpha']])
        save_matrix(func_update_alpha.T,col_name=['alpha_'+str(i) for i in range(1,func_update_alpha.shape[0]+1)],
                   file_name = alpha_fname)
        #only use the alphas with nonzero estimated variance
        indicator = np.array([int(x) for x in unif_d['V']!=0])
        new_func_update_alpha = np.multiply(indicator[:,np.newaxis],func_update_alpha)
        #compute pip
        func_update_pip = 1 - np.prod(1-new_func_update_alpha,axis=0)
        save_array(func_update_pip,col_name = "func_update_pip",file_name=pip_fname)
    return


def compare(trait):
    regions = [os.path.basename(f) for f in glob.glob(RESULT_DIR_ROOT+trait+'/chr*')]
    print('There are {} regions in {} folder'.format(len(regions),trait))
    compare_d = dict()
    susie_func_pips = []
    func_update_pips = []
    for region in regions:
        clear_output(wait=True)
        display("Processing region {}, {} out of {}".format(region,regions.index(region),len(regions)))
        func_d = load_results_d(trait,region,result_type='func')
        if not func_d:
            continue
        alpha_df = pd.read_csv(RESULT_DIR_ROOT+
            '%s/%s/susie_L10_priovar-2.000_func_20bins_minratio100_noninf/func_update_alpha'%(trait,region),
            delim_whitespace=True)
        pip_df = pd.read_csv(RESULT_DIR_ROOT+
            '%s/%s/susie_L10_priovar-2.000_func_20bins_minratio100_noninf/func_update_pip'%(trait,region),
            delim_whitespace=True)
        susie_func_pips.append(func_d['pip'].flatten())
        func_update_pips.append(pip_df.values.flatten())
        compare_d[region] = dict()
        compare_d[region]['alpha_corr'] = stats.pearsonr(alpha_df.values.flatten(),func_d['alpha'].flatten())[0]
        compare_d[region]['pip_corr'] = stats.pearsonr(pip_df.values.flatten(),func_d['pip'].flatten())[0]
        compare_d[region]['top_pip_susie_func'] = func_d['pip'].flatten().max()
        compare_d[region]['top_pip_func_update'] = pip_df.values.flatten().max()
        compare_d[region]['max_diff'] = np.subtract(func_d['pip'].flatten(),pip_df.values.flatten()).max()
        for l in range(1,11):
            alpha_l_corr = stats.pearsonr(alpha_df.iloc[:,l-1].values.flatten(),func_d['alpha'][l-1,:].flatten())[0]
            compare_d[region]['alpha_'+str(l)+'_corr'] = alpha_l_corr
        compare_df = pd.DataFrame(compare_d).T
        compare_df.to_csv(RESULT_DIR_ROOT+trait+'/compare_func_update_susie.csv',sep='\t')
    susie_func_pips = np.array(susie_func_pips)
    func_update_pips = np.array(func_update_pips)
    susie_func_pip = susie_func_pips.flatten()
    func_update_pip = func_update_pips.flatten()
    save_array(susie_func_pip,col_name='SuSiE_func_pip',file_name=RESULT_DIR_ROOT+trait+'/susie_func_all.csv')
    save_array(func_update_pip,col_name='update_func_pip',file_name=RESULT_DIR_ROOT+trait+'/update_func_all.csv')
    return


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trait')
    parser.add_argument('--result-dir')
    parser.add_argument('--compute-func-update',action='store_true')
    parser.add_argument('--compare')
    args = parser.parse_args()

    RESULT_DIR_ROOT = args.result_dir
    if args.compute_func_update:
        compute_func_update(args.trait)
    elif args.compare:
        compare(args.trait)
