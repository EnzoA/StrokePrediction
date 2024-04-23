def split_dataset():
        import awswrangler as wr
        from sklearn.model_selection import train_test_split
        from utils.environment_variables import EnvironmentVariables
        
        dataset = wr.s3.read_csv(EnvironmentVariables.S3_RAW_DATASET.value)
        
        dataset.drop(['id'], axis=1, inplace=True)

        TARGET_COLUMN = 'stroke'

        gender_other_index = dataset[dataset['gender'] == 'Other'].index
        dataset.drop(gender_other_index, axis=0, inplace=True)

        X = dataset.loc[:, dataset.columns != TARGET_COLUMN]
        y = dataset.loc[:, TARGET_COLUMN]

        X_train, X_test, Y_train, Y_test = train_test_split(
            X,
            y,
            stratify=y,
            test_size=0.2,
            random_state=42
        )

        wr.s3.to_csv(X_train, EnvironmentVariables.S3_X_TRAIN.value, index=False)
        wr.s3.to_csv(X_test, EnvironmentVariables.S3_X_TEST.value, index=False)
        wr.s3.to_csv(Y_train, EnvironmentVariables.S3_Y_TRAIN.value, index=False)
        wr.s3.to_csv(Y_test, EnvironmentVariables.S3_Y_TEST.value, index=False)