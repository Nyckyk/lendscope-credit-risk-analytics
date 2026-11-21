
WITH expected AS (

    SELECT
        COALESCE(
            NULLIF(BTRIM(borrower_state), ''),
            'Unknown'
        ) AS borrower_state,

        COUNT(*) AS loan_count,
        SUM(loan_amount) AS total_loan_amount

    FROM {{ ref('stg_loans') }}

    GROUP BY 1

),

actual AS (

    SELECT
        borrower_state,
        loan_count,
        total_loan_amount

    FROM {{ ref('mart_loan_by_state') }}

)

SELECT
    COALESCE(
        e.borrower_state,
        a.borrower_state
    ) AS borrower_state,

    e.loan_count AS expected_count,
    a.loan_count AS actual_count,

    e.total_loan_amount AS expected_amount,
    a.total_loan_amount AS actual_amount

FROM expected e

FULL OUTER JOIN actual a
    ON e.borrower_state = a.borrower_state

WHERE
    e.borrower_state IS NULL
    OR a.borrower_state IS NULL
    OR e.loan_count IS DISTINCT FROM a.loan_count
    OR e.total_loan_amount IS DISTINCT FROM a.total_loan_amount
