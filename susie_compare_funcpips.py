import os
import pandas as pd
import pickle
import glob
import warnings
import gzip

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


def read_unif_dict(trait, region):
    '''
    Return the saved SuSiE R object as a dictionary 
    #' @return a susie fit, which is a list with some or all of the following elements\cr
    #' \item{alpha}{an L by p matrix of posterior inclusion probabilites}
    #' \item{mu}{an L by p matrix of posterior means (conditional on inclusion)}
    #' \item{mu2}{an L by p matrix of posterior second moments (conditional on inclusion)}
    #' \item{XtXr}{an p vector of t(X) times fitted values, the fitted values equal to X times colSums(alpha*mu))}
    #' \item{sigma2}{residual variance}
    #' \item{V}{prior variance}
    '''
    result_unif_file = os.path.join(RESULT_DIR_ROOT,
                                '%s/%s/susie_L10_priovar-2.000_noninf/susie.pickle.gz'%(trait,region))
    unif_dict = read_pickle_dict(result_unif_file)
 
    return unif_dict


def read_pickle_dict(file_name):
    '''
    read FILE.pickle.gz file
    '''
    if os.path.exists(file_name):
        with gzip.GzipFile(file_name,'rb') as f:
             d = pickle.load(f)
        return d
    else:
        warnings.warn('File doesn not exist {}'.format(file_name))
        return dict()


def analytical_update(unif_dict, snpvar):
    return












