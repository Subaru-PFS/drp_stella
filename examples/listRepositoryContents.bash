sqlite3 tigress/HSC/PFS/2015-11-20/registry.sqlite3
.header on
SELECT visit, arm, ccd, count(ccd)
FROM raw
GROUP BY visit, arm, ccd;
