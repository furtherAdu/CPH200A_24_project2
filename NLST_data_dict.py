subgroup_dict = dict(ethnic={1: "Hispanic or Latino",
                             2: "Neither Hispanic nor Latino",
                             7: "Participant refused to answer",
                             95: "Missing data form - form is not expected to ever be completed",
                             98: "Missing - form was submitted and the answer was left blank",
                             99: "Unknown/ decline to answer"},
                     educat={1: "8th grade or less",
                             2: "9th-11th grade",
                             3: "High school graduate/GED",
                             4: "Post high school training, excluding college",
                             5: "Associate degree/ some college",
                             6: "Bachelors Degree",
                             7: "Graduate School",
                             8: "Other",
                             95: "Missing data form - form is not expected to ever be completed",
                             98: "Missing - form was submitted and the answer was left blank",
                             99: "Unknown/ decline to answer"},
                     gender={1: "Male",
                             2: "Female"},
                     race={1: "White",
                           2: "Black or African-American",
                           3: "Asian",
                           4: "American Indian or Alaskan Native",
                           5: "Native Hawaiian or Other Pacific Islander",
                           6: "More than one race",
                           7: "Participant refused to answer",
                           95: "Missing data form - form is not expected to ever be completed",
                           96: "Missing - no response",
                           98: "Missing - form was submitted and the answerwas left blank",
                           99: "Unknown/ decline to answer"}
                     )

feature_transforms = dict(
    histogram=[
        'age'
        ],
    numerical=[
        'pkyr',
        'age',
        ],
    categorical=[
        'gender',
        'race',
        'ethnic',
        'educat',
        ])

clinical_feature_type = dict(
    age='numerical',
    pkyr='numerical',
    gender='categorical'
)

subgroup_feature_type = dict(
    age='histogram',
    race='categorical',
    ethnic='categorical',
    educat='categorical',
    gender='categorical'
)