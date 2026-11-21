
-- Check 1: Employment length categories

SELECT
    COALESCE(
        NULLIF(BTRIM(r.emp_length), ''),
        '(missing)'
    ) AS employment_length,
    COUNT(*) AS loans

FROM staging.stg_loans s

JOIN raw.loans r
    ON r.id = s.loan_id::TEXT

WHERE s.term_months = 36
  AND s.issue_date >= DATE '2011-01-01'
  AND s.issue_date < DATE '2016-01-01'
  AND s.loan_status IN ('Fully Paid', 'Charged Off')

GROUP BY 1
ORDER BY loans DESC;


-- Check 2: Credit history date formats

SELECT
    CASE
        WHEN BTRIM(r.earliest_cr_line)
            ~* '^[a-z]{3}-[0-9]{4}$'
            THEN 'Mon-YYYY'

        WHEN BTRIM(r.earliest_cr_line)
            ~* '^[a-z]{3}-[0-9]{2}$'
            THEN 'Mon-YY'

        ELSE 'Unexpected'
    END AS date_format,

    COUNT(*) AS loans,
    MIN(r.earliest_cr_line) AS example_value

FROM staging.stg_loans s

JOIN raw.loans r
    ON r.id = s.loan_id::TEXT

WHERE s.term_months = 36
  AND s.issue_date >= DATE '2011-01-01'
  AND s.issue_date < DATE '2016-01-01'
  AND s.loan_status IN ('Fully Paid', 'Charged Off')

GROUP BY 1
ORDER BY loans DESC;


-- Check 3: Numeric feature formats and ranges

WITH cohort AS (
    SELECT
        r.revol_util,
        r.inq_last_6mths,
        r.delinq_2yrs,
        r.pub_rec,
        r.open_acc

    FROM staging.stg_loans s

    JOIN raw.loans r
        ON r.id = s.loan_id::TEXT

    WHERE s.term_months = 36
      AND s.issue_date >= DATE '2011-01-01'
      AND s.issue_date < DATE '2016-01-01'
      AND s.loan_status IN ('Fully Paid', 'Charged Off')
),

feature_values AS (
    SELECT
        v.feature,
        NULLIF(BTRIM(v.raw_value), '') AS value

    FROM cohort c

    CROSS JOIN LATERAL (
        VALUES
            ('revol_util', c.revol_util),
            ('inq_last_6mths', c.inq_last_6mths),
            ('delinq_2yrs', c.delinq_2yrs),
            ('pub_rec', c.pub_rec),
            ('open_acc', c.open_acc)
    ) AS v(feature, raw_value)
)

SELECT
    feature,
    COUNT(*) AS loans,

    COUNT(*) FILTER (
        WHERE value IS NULL
    ) AS missing,

    COUNT(*) FILTER (
        WHERE value IS NOT NULL
          AND value !~ '^[0-9]+([.][0-9]+)?$'
    ) AS invalid_format,

    MIN(
        CASE
            WHEN value ~ '^[0-9]+([.][0-9]+)?$'
            THEN value::NUMERIC
        END
    ) AS min_value,

    MAX(
        CASE
            WHEN value ~ '^[0-9]+([.][0-9]+)?$'
            THEN value::NUMERIC
        END
    ) AS max_value

FROM feature_values

GROUP BY feature
ORDER BY feature;
