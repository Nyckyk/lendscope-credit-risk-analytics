
{{ config(
    materialized='table',
    schema='analytics'
) }}

WITH grouped AS (

    SELECT
        EXTRACT(YEAR FROM issue_date)::INTEGER AS issue_year,

        COALESCE(
            NULLIF(BTRIM(loan_purpose), ''),
            'Unknown'
        ) AS loan_purpose,

        COUNT(*) AS loan_count,

        SUM(loan_amount) AS total_loan_amount,

        ROUND(AVG(loan_amount), 2) AS avg_loan_amount,

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

    FROM {{ ref('stg_loans') }}

    GROUP BY 1, 2
)

SELECT
    *,

    ROUND(
        100.0 * total_loan_amount
        / NULLIF(
            SUM(total_loan_amount)
                OVER (PARTITION BY issue_year),
            0
        ),
        4
    ) AS share_of_year_amount_pct,

    ROUND(
        100.0 * charged_off_loans
        / NULLIF(resolved_loans, 0),
        2
    ) AS charged_off_share_resolved_pct

FROM grouped
