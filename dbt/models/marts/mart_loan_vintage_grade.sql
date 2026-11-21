{{ config(
    materialized='table',
    schema='analytics'
) }}

WITH loans AS (

    SELECT
        EXTRACT(YEAR FROM issue_date)::INTEGER AS issue_year,
        grade,
        loan_amount,
        interest_rate_pct,
        loan_status

    FROM {{ ref('stg_loans') }}

),

grouped AS (

    SELECT
        issue_year,
        grade,

        COUNT(*) AS loan_count,

        SUM(loan_amount) AS total_loan_amount,

        ROUND(AVG(interest_rate_pct), 2)
            AS avg_interest_rate_pct,

        COUNT(*) FILTER (
            WHERE loan_status = 'Fully Paid'
        ) AS fully_paid_loans,

        COUNT(*) FILTER (
            WHERE loan_status = 'Charged Off'
        ) AS charged_off_loans,

        COUNT(*) FILTER (
            WHERE loan_status = 'Current'
        ) AS current_loans,

        COUNT(*) FILTER (
            WHERE loan_status IN ('Fully Paid', 'Charged Off')
        ) AS resolved_loans

    FROM loans
    GROUP BY issue_year, grade

)

SELECT
    *,

    loan_count - resolved_loans AS other_status_loans,

    ROUND(
        100.0 * charged_off_loans
        / NULLIF(resolved_loans, 0),
        2
    ) AS charged_off_share_resolved_pct

FROM grouped