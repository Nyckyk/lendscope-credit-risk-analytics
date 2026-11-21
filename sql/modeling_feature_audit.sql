
-- Check 1: Confirm candidate features exist in raw loans.

SELECT
    column_name,
    data_type
FROM information_schema.columns
WHERE table_schema = 'raw'
  AND table_name = 'loans'
  AND column_name IN (
      'emp_length',
      'revol_util',
      'inq_last_6mths',
      'delinq_2yrs',
      'pub_rec',
      'earliest_cr_line',
      'open_acc',
      'verification_status'
  )
ORDER BY column_name;


-- Check 2: Measure missing values in the modelling cohort.

WITH cohort AS (
    SELECT
        s.loan_id,
        r.emp_length,
        r.revol_util,
        r.inq_last_6mths,
        r.delinq_2yrs,
        r.pub_rec,
        r.earliest_cr_line,
        r.open_acc,
        r.verification_status

    FROM staging.stg_loans AS s

    INNER JOIN raw.loans AS r
        ON r.id = s.loan_id::TEXT

    WHERE s.term_months = 36
      AND s.issue_date >= DATE '2011-01-01'
      AND s.issue_date < DATE '2016-01-01'
      AND s.loan_status IN (
          'Fully Paid',
          'Charged Off'
      )
)

SELECT
    COUNT(*) AS total_loans,

    COUNT(*) FILTER (
        WHERE NULLIF(BTRIM(emp_length), '') IS NULL
    ) AS missing_employment_length,

    COUNT(*) FILTER (
        WHERE NULLIF(BTRIM(revol_util), '') IS NULL
    ) AS missing_revolving_utilisation,

    COUNT(*) FILTER (
        WHERE NULLIF(BTRIM(inq_last_6mths), '') IS NULL
    ) AS missing_recent_inquiries,

    COUNT(*) FILTER (
        WHERE NULLIF(BTRIM(delinq_2yrs), '') IS NULL
    ) AS missing_delinquencies,

    COUNT(*) FILTER (
        WHERE NULLIF(BTRIM(pub_rec), '') IS NULL
    ) AS missing_public_records,

    COUNT(*) FILTER (
        WHERE NULLIF(BTRIM(earliest_cr_line), '') IS NULL
    ) AS missing_earliest_credit_line,

    COUNT(*) FILTER (
        WHERE NULLIF(BTRIM(open_acc), '') IS NULL
    ) AS missing_open_accounts,

    COUNT(*) FILTER (
        WHERE NULLIF(BTRIM(verification_status), '') IS NULL
    ) AS missing_verification_status,

    COUNT(*) FILTER (
        WHERE NULLIF(BTRIM(revol_util), '') IS NOT NULL
          AND BTRIM(revol_util)
              !~ '^[0-9]+([.][0-9]+)?$'
    ) AS unexpected_utilisation_formats

FROM cohort;
