
{{ config(
    materialized='table',
    schema='analytics'
) }}

WITH state_summary AS (

    SELECT
        COALESCE(
            NULLIF(BTRIM(borrower_state), ''),
            'Unknown'
        ) AS borrower_state,

        COUNT(*) AS loan_count,

        SUM(loan_amount) AS total_loan_amount,

        ROUND(AVG(loan_amount), 2)
            AS avg_loan_amount,

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
            WHERE loan_status IN (
                'Fully Paid', 'Charged Off'
            )
        ) AS resolved_loans

    FROM {{ ref('stg_loans') }}
    GROUP BY 1

),

portfolio_total AS (

    SELECT
        SUM(total_loan_amount)
            AS portfolio_loan_amount
    FROM state_summary

)

SELECT
    s.*,

    ROUND(
        100.0 * s.total_loan_amount /
        NULLIF(p.portfolio_loan_amount, 0),
        2
    ) AS origination_amount_share_pct,

    ROUND(
        100.0 * s.charged_off_loans /
        NULLIF(s.resolved_loans, 0),
        2
    ) AS charged_off_share_resolved_pct

FROM state_summary s
CROSS JOIN portfolio_total p
