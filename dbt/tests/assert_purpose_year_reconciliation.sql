
WITH staging_totals AS (

    SELECT
        COUNT(*) AS loan_count,
        SUM(loan_amount) AS total_amount
    FROM {{ ref('stg_loans') }}

),

mart_totals AS (

    SELECT
        SUM(loan_count) AS loan_count,
        SUM(total_loan_amount) AS total_amount
    FROM {{ ref('mart_loan_purpose_year') }}

)

SELECT
    s.loan_count AS staging_count,
    m.loan_count AS mart_count,
    s.total_amount AS staging_amount,
    m.total_amount AS mart_amount

FROM staging_totals s
CROSS JOIN mart_totals m

WHERE
    s.loan_count IS DISTINCT FROM m.loan_count
    OR s.total_amount IS DISTINCT FROM m.total_amount
