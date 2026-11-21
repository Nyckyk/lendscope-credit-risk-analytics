
WITH yearly_outcomes AS (

    SELECT
        EXTRACT(YEAR FROM issue_date)::INTEGER
            AS issue_year,

        COUNT(*) AS total_loans,

        COUNT(*) FILTER (
            WHERE loan_status = 'Fully Paid'
        ) AS fully_paid,

        COUNT(*) FILTER (
            WHERE loan_status = 'Charged Off'
        ) AS charged_off,

        COUNT(*) FILTER (
            WHERE loan_status NOT IN (
                'Fully Paid',
                'Charged Off'
            )
        ) AS other_status,

        COUNT(*) FILTER (
            WHERE loan_status IN (
                'Fully Paid',
                'Charged Off'
            )
        ) AS resolved_loans

    FROM staging.stg_loans

    WHERE term_months = 36
      AND issue_date >= DATE '2009-01-01'
      AND issue_date < DATE '2017-01-01'

    GROUP BY 1
)

SELECT
    issue_year,
    total_loans,
    fully_paid,
    charged_off,
    other_status,

    ROUND(
        100.0 * resolved_loans
        / NULLIF(total_loans, 0),
        2
    ) AS resolved_pct

FROM yearly_outcomes

ORDER BY issue_year;
