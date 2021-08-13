"""fishdata app"""


import streamlit as st
import pandas as pd
import numpy as np
from tableone import TableOne

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import SessionState


#define function to return column indecies for each column name passed
def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]


def main():
    st.title("Fish Data Analysis")

    #session state variables
    session_state = SessionState.get(tidy=None)

    menu = ["Welcome", "Upload Data", "Clustering"]
    choice = st.sidebar.selectbox("Menu", menu)

    #display page choice
    if choice == menu[0]:
        #welcome page
        st.subheader(menu[0])
    elif choice == menu[1]:
        #processing page
        st.subheader(menu[1])
        #upload data
        data_file = st.file_uploader("Upload csv file", type=["csv"])
        #process data
        if st.button("Process File") and (data_file is not None):
            #load data as dataframe
            df = pd.read_csv(data_file)
            #save headers in row 0
            headers = df.loc[0]
            #display unique headers
            feats = headers.dropna().str.strip().unique()
            #st.write(feats)
            #select covariates
            covariates = feats[1:3]
            #st.write("Covariates: " + covariates)
            #select feature degrees
            degs = feats[4:]
            #st.write("feature degrees: " + degs)
            #drop headers in row 0 from df
            df.drop(index=df.index[0], axis=0, inplace=True)
            #remove white space in IDs in column 0
            df[df.columns[0]] = df[df.columns[0]].str.strip()
            #convert all data to numeric
            df = df.apply(pd.to_numeric, errors='coerce')
            #drop rows where element in column 0 is not a number
            df.dropna(axis=0, subset=[df.columns[0]], inplace=True)
            #display dataframe
            #st.dataframe(df)
            #display unique features
            feats = df.loc[:, ~df.columns.str.contains('^Unnamed')].columns
            #st.write(feats)

            #create dataframe to hold tidy data
            #st.write(str([*covariates, *feats]))
            tidy = pd.DataFrame(index=df[df.columns[0]], columns=[*covariates, *feats])
            #st.dataframe(tidy)

            #add Covariates
            tidy[covariates] = df[df.columns[1:3]]

            # get column indecies in df for each feature
            featidx = column_index(df, feats)
            #st.write(featidx)

            #add feature degrees
            for row in range(len(tidy)):
                #loop over features
                for ifeat, col in enumerate(featidx):
                    #search the next ndegree columns to get the feature degree
                    deg = np.argmax(df.iloc[row, col:(col+len(degs))])
                    #assign degree value
                    tidy.iloc[row, ifeat+len(covariates)] = deg

            #replace any missing values with missing
            tidy[tidy<0] = np.nan

            #display the tidy data
            st.info("Formatted Data")
            st.dataframe(tidy)

            #generate table 1
            tbl1 = TableOne(tidy, categorical = [feat for feat in feats], htest_name=True,
                            dip_test=True, normal_test=True, tukey_test=True)
            st.info("Formatted Data Stats")
            st.markdown(tbl1.tabulate(tablefmt = "html"), unsafe_allow_html=True)

            #impute missing using Iterative imputer
            #use mean esitmator for numerical covariates
            imp_num = IterativeImputer(estimator=RandomForestRegressor(),
                                       initial_strategy='mean',
                                       max_iter=10, random_state=0)
            #use most-frequent estimator for categorical features
            imp_cat = IterativeImputer(estimator=RandomForestClassifier(),
                                       initial_strategy='most_frequent',
                                       max_iter=10, random_state=0)

            tidy[covariates] = imp_num.fit_transform(tidy[covariates])
            tidy[feats] = imp_cat.fit_transform(tidy[feats])

            #save tidy data in session data
            session_state.tidy = tidy

            #display the tidy data
            st.info("Imputed Data")
            st.dataframe(tidy)

            #generate table 2
            tbl2 = TableOne(tidy, categorical = [feat for feat in feats], htest_name=True,
                            dip_test=True, normal_test=True, tukey_test=True)
            st.info("Imputed Data Stats")
            st.markdown(tbl2.tabulate(tablefmt = "html"), unsafe_allow_html=True)


    else:
        #analysis page
        st.subheader(menu[-1])

        #params form
        form = st.sidebar.form(key='Clustering_params')
        min_n = form.slider("min_n", min_value=2, max_value=20, value=10, step=1)
        eps = form.slider("eps", min_value=.01, max_value=10.0, value=.3, step=0.01)
        xfeat = form.selectbox("Select feature to plot on x axis",
            session_state.tidy.columns)
        yfeat = form.selectbox("Select feature to plot on y axis",
            session_state.tidy.columns)
        submit_button = form.form_submit_button(label='Submit')

        if submit_button and (session_state.tidy is not None):
            st.info("DBSCAN Clustering Results.")
            #DBSCAN clustering
            X = np.array(session_state.tidy)
            #X = StandardScaler().fit_transform(X)
            db = DBSCAN(eps=0.5, min_samples=5).fit(X)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_

            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            st.write('Estimated number of clusters: %d' % n_clusters_)
            st.write('Estimated number of noise points: %d' % n_noise_)
            #st.write("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
            #st.write("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
            #st.write("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
            #st.write("Adjusted Rand Index: %0.3f"
            #      % metrics.adjusted_rand_score(labels_true, labels))
            #st.write("Adjusted Mutual Information: %0.3f"
            #      % metrics.adjusted_mutual_info_score(labels_true, labels))
            if n_clusters_ >1:
                st.write("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

            # Black removed and is used for noise instead.
            unique_labels = set(labels)
            colors = [plt.cm.Spectral(each)
                      for each in np.linspace(0, 1, len(unique_labels))]
            fig, ax = plt.subplots()
            fidx = column_index(session_state.tidy, [xfeat, yfeat])
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]

                class_member_mask = (labels == k)

                xy = X[class_member_mask & core_samples_mask]
                ax.plot(xy[:, fidx[0]], xy[:, fidx[1]], 'o', markerfacecolor=tuple(col),
                         markeredgecolor='k', markersize=14)

                xy = X[class_member_mask & ~core_samples_mask]
                ax.plot(xy[:, fidx[0]], xy[:, fidx[1]], 'o', markerfacecolor=tuple(col),
                         markeredgecolor='k', markersize=6)

            ax.set_title('Estimated number of clusters: %d' % n_clusters_)
            ax.set_xlabel(xfeat)
            ax.set_ylabel(yfeat)
            # Plot figure
            st.pyplot(fig)

            #tsne plot
            st.info("t-SNE Embedding")
            X = StandardScaler().fit_transform(X)
            X_embedded = TSNE(n_components=2).fit_transform(X)
            fig, ax = plt.subplots()
            ax.scatter(X_embedded[:, 0], X_embedded[:, 1])
            for i in range(len(X_embedded[:, 0])):
                ax.annotate(str(i), (X_embedded[i, 0], X_embedded[i, 1]))
            st.pyplot(fig)


if __name__ == '__main__':
    main()
