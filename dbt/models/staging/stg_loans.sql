
WITH source_loans AS (

    SELECT *
    FROM {{ source('lendingclub', 'loans') }}
    WHERE id ~ '^[0-9]+$'

)

SELECT
    BTRIM(id) AS loan_id,

    NULLIF(BTRIM(loan_amnt), '')::NUMERIC
        AS loan_amount,

    NULLIF(BTRIM(funded_amnt), '')::NUMERIC
        AS funded_amount,

    NULLIF(
        REGEXP_REPLACE(term, '[^0-9]', '', 'g'),
        ''
    )::INTEGER AS term_months,

    NULLIF(BTRIM(int_rate), '')::NUMERIC
        AS interest_rate_pct,

    NULLIF(BTRIM(installment), '')::NUMERIC
        AS monthly_installment,

    NULLIF(BTRIM(annual_inc), '')::NUMERIC
        AS annual_income,

    NULLIF(BTRIM(dti), '')::NUMERIC
        AS debt_to_income_ratio,

    NULLIF(BTRIM(fico_range_low), '')::NUMERIC::INTEGER
    AS fico_range_low,

    NULLIF(BTRIM(fico_range_high), '')::NUMERIC::INTEGER
    AS fico_range_high,

    NULLIF(BTRIM(grade), '') AS grade,
    NULLIF(BTRIM(sub_grade), '') AS sub_grade,
    NULLIF(BTRIM(purpose), '') AS loan_purpose,
    NULLIF(BTRIM(home_ownership), '') AS home_ownership,
    NULLIF(BTRIM(addr_state), '') AS borrower_state,

    TO_DATE(BTRIM(issue_d), 'Mon-YYYY')
        AS issue_date,

    NULLIF(BTRIM(loan_status), '') AS loan_status

FROM source_loans
