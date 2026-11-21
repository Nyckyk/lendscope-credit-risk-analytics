
WITH staging_count AS (

    SELECT COUNT(*) AS total
    FROM {{ ref('stg_loans') }}

),

mart_count AS (

    SELECT COALESCE(SUM(loan_count), 0) AS total
    FROM {{ ref('mart_loan_vintage_grade') }}

)

SELECT
    staging_count.total AS staging_total,
    mart_count.total AS mart_total

FROM staging_count
CROSS JOIN mart_count

WHERE staging_count.total
    IS DISTINCT FROM mart_count.total
