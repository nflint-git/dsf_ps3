# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask_ml.preprocessing import Categorizer
from glum import GeneralizedLinearRegressor, TweedieDistribution
from lightgbm import LGBMRegressor, plot_metric
from sklearn.compose import ColumnTransformer
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from ps3.data import create_sample_split, load_transform

# %%
# load data
df = load_transform()

# %%
# Train benchmark tweedie model. This is entirely based on the glum tutorial.
weight = df["Exposure"].values
df["PurePremium"] = df["ClaimAmountCut"] / df["Exposure"]
y = df["PurePremium"]
# TODO: Why do you think, we divide by exposure here to arrive at our outcome variable?
# We divide by exposure because we want the total claim amount per exposure unit per policy

#%%
df.head()
# %%
# TODO: use your create_sample_split function here
df = create_sample_split(df, id_column='IDpol')
train = np.where(df["sample"] == "train")
test = np.where(df["sample"] == "test")
df_train = df.iloc[train].copy()
df_test = df.iloc[test].copy()

categoricals = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge", "VehPower"]

predictors = categoricals + ["BonusMalus", "Density"]
glm_categorizer = Categorizer(columns=categoricals)

X_train_t = glm_categorizer.fit_transform(df[predictors].iloc[train])
X_test_t = glm_categorizer.transform(df[predictors].iloc[test])
y_train_t, y_test_t = y.iloc[train], y.iloc[test]
w_train_t, w_test_t = weight[train], weight[test]

TweedieDist = TweedieDistribution(1.5)
t_glm1 = GeneralizedLinearRegressor(family=TweedieDist, l1_ratio=1, fit_intercept=True)
t_glm1.fit(X_train_t, y_train_t, sample_weight=w_train_t)


pd.DataFrame(
    {"coefficient": np.concatenate(([t_glm1.intercept_], t_glm1.coef_))},
    index=["intercept"] + t_glm1.feature_names_,
).T

df_test["pp_t_glm1"] = t_glm1.predict(X_test_t)
df_train["pp_t_glm1"] = t_glm1.predict(X_train_t)

print(
    "training loss t_glm1:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm1"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm1:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm1"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * t_glm1.predict(X_test_t)),
    )
)
# %%
# TODO: Let's add splines for BonusMalus and Density and use a Pipeline.
# Steps: 
# 1. Define a Pipeline which chains a StandardScaler and SplineTransformer. 
#    Choose knots="quantile" for the SplineTransformer and make sure, we 
#    are only including one intercept in the final GLM. 
# 2. Put the transforms together into a ColumnTransformer. Here we use OneHotEncoder for the categoricals.
# 3. Chain the transforms together with the GLM in a Pipeline.

# Let's put together a pipeline
numeric_cols = ["BonusMalus", "Density"]
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('spline', SplineTransformer(knots= "quantile"))
    ])

preprocessor = ColumnTransformer(
    transformers=[
        # TODO: Add numeric transforms here
        ("num", numeric_transformer, numeric_cols),
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), categoricals)
    ]
)
preprocessor.set_output(transform="pandas")
t_glm2 = GeneralizedLinearRegressor(family=TweedieDist, l1_ratio=1, fit_intercept=True)
model_pipeline = Pipeline(
    # TODO: Define pipeline steps here
    steps = [("preproccesor", preprocessor),
             ("model", t_glm2)]
)

#%%
# let's have a look at the pipeline
model_pipeline

#%%
# let's check that the transforms worked
z = model_pipeline[:-1].fit_transform(df_train)
print(z)
model_pipeline.fit(df_train, y_train_t, model__sample_weight=w_train_t)

#%%
pd.DataFrame(
    {
        "coefficient": np.concatenate(
            ([model_pipeline[-1].intercept_], model_pipeline[-1].coef_)
        )
    },
    index=["intercept"] + model_pipeline[-1].feature_names_,
).T

df_test["pp_t_glm2"] = model_pipeline.predict(df_test)
df_train["pp_t_glm2"] = model_pipeline.predict(df_train)

print(
    "training loss t_glm2:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm2"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm2:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm2"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_glm2"]),
    )
)

# %%
# TODO: Let's use a GBM instead as an estimator.
# Steps
# 1: Define the modelling pipeline. Tip: This can simply be a LGBMRegressor based on X_train_t from before.
# 2. Make sure we are choosing the correct objective for our estimator.

lgbm_1 = LGBMRegressor(objective = "tweedie", tweedie_variance_power = 1.5)

model_pipeline = Pipeline(
    # TODO: Define pipeline steps here
    steps = [("preproccesor", preprocessor),
             ("model", lgbm_1)])

model_pipeline.fit(X_train_t, y_train_t, model__sample_weight=w_train_t)
df_test["pp_t_lgbm"] = model_pipeline.predict(X_test_t)
df_train["pp_t_lgbm"] = model_pipeline.predict(X_train_t)
print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

# %%
# TODO: Let's tune the LGBM to reduce overfitting.
# Steps:
# 1. Define a `GridSearchCV` object with our lgbm pipeline/estimator. Tip: Parameters for a specific step of the pipeline
# can be passed by <step_name>__param. 

# Note: Typically we tune many more parameters and larger grids,
# but to save compute time here, we focus on getting the learning rate
# and the number of estimators somewhat aligned -> tune learning_rate and n_estimators
param_grid = {
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__n_estimators': [50, 100, 200]

}
cv = GridSearchCV(estimator=model_pipeline, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
cv.fit(X_train_t, y_train_t, model__sample_weight=w_train_t)

df_test["pp_t_lgbm"] = cv.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm"] = cv.best_estimator_.predict(X_train_t)

print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_lgbm"]),
    )
)
# %%
# Let's compare the sorting of the pure premium predictions


# Source: https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
def lorenz_curve(y_true, y_pred, exposure):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
    return cumulated_samples, cumulated_claim_amount


fig, ax = plt.subplots(figsize=(8, 8))

for label, y_pred in [
    ("LGBM", df_test["pp_t_lgbm"]),
    ("GLM Benchmark", df_test["pp_t_glm1"]),
    ("GLM Splines", df_test["pp_t_glm2"]),
]:
    ordered_samples, cum_claims = lorenz_curve(
        df_test["PurePremium"], y_pred, df_test["Exposure"]
    )
    gini = 1 - 2 * auc(ordered_samples, cum_claims)
    label += f" (Gini index: {gini: .3f})"
    ax.plot(ordered_samples, cum_claims, linestyle="-", label=label)

# Oracle model: y_pred == y_test
ordered_samples, cum_claims = lorenz_curve(
    df_test["PurePremium"], df_test["PurePremium"], df_test["Exposure"]
)
gini = 1 - 2 * auc(ordered_samples, cum_claims)
label = f"Oracle (Gini index: {gini: .3f})"
ax.plot(ordered_samples, cum_claims, linestyle="-.", color="gray", label=label)

# Random baseline
ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random baseline")
ax.set(
    title="Lorenz Curves",
    xlabel="Fraction of policyholders\n(ordered by model from safest to riskiest)",
    ylabel="Fraction of total claim amount",
)
ax.legend(loc="upper left")
plt.plot()

# %%
## START OF PS 4

## Task 1 - train a constrained LGBM by introducing a monotonicity constraint for BonusMalus and LGBMRegressor, CV this and compare the prediction of constrained vs Unconstrained

# Create a plot of the average claims per BonusMalus group, weight them by exposure. What will/could happen if we do not include a monotonicity constraint?
# Create a plot of the average claims per BonusMalus group, weighted by exposure
avg_claims_per_group = df.groupby("BonusMalus").apply(
    lambda x: np.average(x["PurePremium"], weights=x["Exposure"])
)

plt.figure(figsize=(10, 6))
plt.plot(avg_claims_per_group.index, avg_claims_per_group.values, marker='o')
plt.xlabel("BonusMalus")
plt.ylabel("Average Claims (Weighted by Exposure)")
plt.title("Average Claims per BonusMalus Group")
plt.grid(True)
plt.show()

# %%
# Create a new model pipeline or estimator called constraine_lgbm, introduce a monotonicity constraint for BonusMalus.
# Define the model pipeline
monotone_constraints = [1] * 7 + [0] * (64 - 7) #note that this needs to be of the size for after prepocessing
constrained_lgbm = LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5, mc = monotone_constraints,  monotone_constraints_method="basic")
model_pipeline = Pipeline(
    # TODO: Define pipeline steps here
    steps = [("preprocessor", preprocessor),
             ("model", constrained_lgbm)])

model_pipeline.fit(X_train_t, y_train_t, model__sample_weight=w_train_t)
# %%
# Cross validate and predict using the es estimator, save the predictions in pp_t_lgbm_constrained
cv = GridSearchCV(estimator=model_pipeline, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
cv.fit(X_train_t, y_train_t, model__sample_weight=w_train_t)

df_train["pp_t_lgbm_constrained"] = cv.best_estimator_.predict(X_train_t)
df_test["pp_t_lgbm_constrained"] = cv.best_estimator_.predict(X_test_t)


print(
    "training loss t_lgbm_constrained:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm_constrained"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm_constrained:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm_constrained"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_lgbm_constrained"]),
    )
)

# compare the training loss results
'''
tuned unconstrained lgbm
training loss t_lgbm:  67.99717663819501
testing loss t_lgbm:  73.83532107032258

tuned constrained
training loss t_lgbm_constrained:  69.10954624624668
testing loss t_lgbm_constrained:  74.16978686840706

The constrained model has slightly higher training loss compared to the unconstrained model. 
This is expected because the monotonic constraints restrict the model's flexibility, preventing it from fully optimizing for the training data.
'''
# %%
# Task 2 - based upon the cv constrained optimizer, plot a learning curve which shows the convergence of the score on the train and test set
best_params = {'learning_rate': 0.1, 'n_estimators': 50} #from previous cv
monotone_constraints = [0,0,0,0,0,0,0,1,0]
eval_constrained_lgbm = LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5, monotone_constraints=monotone_constraints, monotone_constraints_method="basic", **best_params)

eval_constrained_lgbm.fit(X_train_t, y_train_t, sample_weight=w_train_t, eval_set=[(X_train_t, y_train_t), (X_test_t, y_test_t)], eval_names=["train", "test"], eval_metric='l2')

# Plot learning curve using LightGBM's built-in function
plot_metric(eval_constrained_lgbm, metric='l2')
plt.title("Learning Curve")
plt.show()

# %%
best_params = {'learning_rate': 0.1, 'n_estimators': 50} #from previous cv
monotone_constraints = [1] * 7 + [0] * (64 - 7) #note that this needs to be of the size for after prepocessing
eval_set = [(X_train_t, y_train_t), (X_test_t, y_test_t)]
constrained_lgbm = LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5, monotone_constraints=monotone_constraints, monotone_constraints_method="basic", **best_params)
model_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor),
           ("model", constrained_lgbm)])

model_pipeline.fit(X_train_t, y_train_t, model__sample_weight=w_train_t, model__eval_set = eval_set, model__eval_metric= 'l2')

plot_metric(constrained_lgbm, metric='l2')
plt.title("Learning Curve")
plt.show()
# %%
