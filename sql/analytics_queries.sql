-- Reference IR-style queries against the Student Success Analytics warehouse.
-- Run these in DuckDB after `python etl/load_warehouse.py`.

-- 1. Year-over-year retention by cohort
SELECT
    cohort_year,
    COUNT(*)                                         AS cohort_n,
    SUM(returned_year_2)                             AS retained_n,
    ROUND(AVG(returned_year_2) * 100, 1)             AS retention_rate_pct
FROM fact_retention
GROUP BY cohort_year
ORDER BY cohort_year;

-- 2. Retention by college and program
SELECT
    p.college,
    p.program_name,
    COUNT(*)                                         AS cohort_n,
    ROUND(AVG(r.returned_year_2) * 100, 1)           AS retention_rate_pct
FROM fact_retention r
JOIN dim_student s USING (student_id)
JOIN dim_program p USING (program_code)
GROUP BY p.college, p.program_name
ORDER BY retention_rate_pct DESC;

-- 3. Retention gap by first-generation status
SELECT
    s.first_generation,
    COUNT(*)                                         AS cohort_n,
    ROUND(AVG(r.returned_year_2) * 100, 1)           AS retention_rate_pct
FROM fact_retention r
JOIN dim_student s USING (student_id)
GROUP BY s.first_generation
ORDER BY s.first_generation;

-- 4. First-term GPA distribution by Pell eligibility
SELECT
    s.pell_eligible,
    ROUND(AVG(t.term_gpa), 3)                        AS avg_term_gpa,
    ROUND(MEDIAN(t.term_gpa), 3)                     AS median_term_gpa,
    ROUND(STDDEV(t.term_gpa), 3)                     AS stddev_term_gpa
FROM fact_term_enrollment t
JOIN dim_student s USING (student_id)
WHERE t.term_seq = 1
GROUP BY s.pell_eligible;

-- 5. Engagement vs. retention -- LMS logins quartile
WITH eng AS (
    SELECT
        student_id,
        AVG(lms_logins_per_week) AS mean_lms
    FROM fact_term_enrollment
    GROUP BY student_id
),
quartiles AS (
    SELECT
        student_id,
        NTILE(4) OVER (ORDER BY mean_lms) AS lms_quartile
    FROM eng
)
SELECT
    q.lms_quartile,
    COUNT(*)                                         AS n,
    ROUND(AVG(r.returned_year_2) * 100, 1)           AS retention_rate_pct
FROM quartiles q
JOIN fact_retention r USING (student_id)
GROUP BY q.lms_quartile
ORDER BY q.lms_quartile;

-- 6. Course evaluation rating distribution by program
SELECT
    p.program_code,
    p.program_name,
    COUNT(*)                                         AS n_evals,
    ROUND(AVG(e.rating), 2)                          AS avg_rating
FROM fact_course_evaluation e
JOIN dim_student s USING (student_id)
JOIN dim_program p USING (program_code)
GROUP BY p.program_code, p.program_name
ORDER BY avg_rating DESC;

-- 7. Free-text presence by demographic (signal that survey response varies)
SELECT
    s.first_generation,
    s.pell_eligible,
    COUNT(DISTINCT s.student_id)                     AS students,
    COUNT(*)                                         AS evaluations,
    ROUND(AVG(e.rating), 2)                          AS avg_rating
FROM fact_course_evaluation e
JOIN dim_student s USING (student_id)
GROUP BY s.first_generation, s.pell_eligible
ORDER BY s.first_generation, s.pell_eligible;

-- 8. Unmet financial need vs. retention
SELECT
    CASE
        WHEN unmet_need < 5000  THEN '0-5k'
        WHEN unmet_need < 15000 THEN '5-15k'
        WHEN unmet_need < 25000 THEN '15-25k'
        ELSE '25k+'
    END                                              AS unmet_need_band,
    COUNT(*)                                         AS n,
    ROUND(AVG(r.returned_year_2) * 100, 1)           AS retention_rate_pct
FROM fact_retention r
JOIN dim_student s USING (student_id)
GROUP BY 1
ORDER BY MIN(unmet_need);
