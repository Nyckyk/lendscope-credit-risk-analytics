
-- Check 1: Investigate unresolved 2015 loans

SELECT
    loan_status,
    COUNT(*) AS loan_count

FROM staging.stg_loans

WHERE term_months = 36
  AND issue_date >= DATE '2015-01-01'
  AND issue_date < DATE '2016-01-01'
  AND loan_status NOT IN ('Fully Paid', 'Charged Off')

GROUP BY loan_status
ORDER BY loan_count DESC;


-- Check 2: Evaluate modelling populations and missing features

WITH candidate_loans AS (

    SELECT
        EXTRACT(YEAR FROM issue_date)::INTEGER AS issue_year,

        CASE
            WHEN loan_status = 'Charged Off' THEN 1
            WHEN loan_status = 'Fully Paid' THEN 0
            ELSE NULL
        END AS target,

        loan_amount,
        annual_income,
        debt_to_income_ratio,
        fico_range_low,
        fico_range_high,
        loan_purpose,
        home_ownership

    FROM staging.stg_loans

    WHERE term_months = 36
      AND issue_date >= DATE '2011-01-01'
      AND issue_date < DATE '2016-01-01'

)

SELECT
    CASE
        WHEN issue_year <= 2013 THEN '1 Training'
        WHEN issue_year = 2014 THEN '2 Validation'
        WHEN issue_year = 2015 THEN '3 Testing'
    END AS dataset_split,

    COUNT(*) AS total_loans,

    COUNT(target) AS labelled_loans,

    COUNT(*) FILTER (
        WHERE target IS NULL
    ) AS unlabelled_loans,

    COUNT(*) FILTER (
        WHERE target = 1
    ) AS charged_off_loans,

    ROUND(
        100.0 * COUNT(*) FILTER (WHERE target = 1)
        / NULLIF(COUNT(target), 0),
        2
    ) AS observed_charge_off_pct,

    COUNT(*) FILTER (
        WHERE target IS NOT NULL
          AND (
              loan_amount IS NULL
              OR annual_income IS NULL
              OR debt_to_income_ratio IS NULL
              OR fico_range_low IS NULL
              OR fico_range_high IS NULL
              OR loan_purpose IS NULL
              OR home_ownership IS NULL
          )
    ) AS labelled_loans_with_missing_features

FROM candidate_loans

GROUP BY 1
ORDER BY 1;
