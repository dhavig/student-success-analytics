-- Star schema for the Student Success Analytics warehouse.
-- DuckDB dialect. Designed for analytic queries and dashboard semantic layers.

DROP TABLE IF EXISTS fact_term_enrollment;
DROP TABLE IF EXISTS fact_retention;
DROP TABLE IF EXISTS fact_course_evaluation;
DROP TABLE IF EXISTS dim_student;
DROP TABLE IF EXISTS dim_program;
DROP TABLE IF EXISTS dim_term;

-- =========================================================================
-- Dimensions
-- =========================================================================

CREATE TABLE dim_student (
    student_id              VARCHAR PRIMARY KEY,
    cohort_year             INTEGER NOT NULL,
    gender                  VARCHAR,
    race_ethnicity          VARCHAR,
    first_generation        VARCHAR,
    pell_eligible           VARCHAR,
    hs_gpa                  DOUBLE,
    sat_total               INTEGER,
    program_code            VARCHAR,
    institutional_aid       DOUBLE,
    federal_aid             DOUBLE,
    unmet_need              DOUBLE,
    distance_from_home_mi   INTEGER
);

CREATE TABLE dim_program (
    program_code    VARCHAR PRIMARY KEY,
    program_name    VARCHAR NOT NULL,
    college         VARCHAR NOT NULL
);

CREATE TABLE dim_term (
    term_key        VARCHAR PRIMARY KEY,   -- e.g. '2023-Fall'
    term_year       INTEGER NOT NULL,
    term_season     VARCHAR NOT NULL,
    term_seq        INTEGER NOT NULL       -- 1=fall of cohort year, 2=spring
);

-- =========================================================================
-- Facts
-- =========================================================================

CREATE TABLE fact_term_enrollment (
    student_id              VARCHAR NOT NULL,
    term_key                VARCHAR NOT NULL,
    term_year               INTEGER NOT NULL,
    term_season             VARCHAR NOT NULL,
    term_seq                INTEGER NOT NULL,
    term_gpa                DOUBLE,
    credits_attempted       INTEGER,
    credits_earned          INTEGER,
    lms_logins_per_week     INTEGER,
    advising_meetings       INTEGER,
    on_campus_housing       INTEGER
);

CREATE TABLE fact_retention (
    student_id                  VARCHAR PRIMARY KEY,
    cohort_year                 INTEGER NOT NULL,
    retention_probability_true  DOUBLE,
    returned_year_2             INTEGER NOT NULL
);

-- Free-text course evaluations -- the "survey + unstructured text" layer.
CREATE TABLE fact_course_evaluation (
    student_id          VARCHAR NOT NULL,
    term_year           INTEGER NOT NULL,
    term_season         VARCHAR NOT NULL,
    course_code         VARCHAR NOT NULL,
    rating              INTEGER NOT NULL,
    comment             VARCHAR NOT NULL
);
