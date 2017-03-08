#!/usr/bin/env python
import argparse
import os
import sys

try:
    import sqlite3 as sqlite
except ImportError:
    import sqlite
try:
    import psycopg2 as pgsql
    havePgSql = True
except ImportError:
    try:
        from pg8000 import DBAPI as pgsql
        havePgSql = True
    except ImportError:
        havePgSql = False
if havePgSql:
    from lsst.daf.butlerUtils import PgSqlConfig

def formatVisits(visits):
    """Format a set of visits into the format used for an --id argument"""
    visits = sorted(visits)

    visitSummary = []
    i = 0
    while i < len(visits):
        v0 = -1

        while i < len(visits):
            v = visits[i]; i += 1
            if v0 < 0:
                v0 = v
                dv = -1                 # visit stride
                continue

            if dv < 0:
                dv = v - v0

            if visits[i - 2] + dv != v:
                i -= 1                  # process this visit again later
                v = visits[i - 1]       # previous value of v
                break

        if v0 == v:
            vstr = "%d" % v
        else:
            if v == v0 + dv:
                vstr = "%d^%d" % (v0, v)
            else:
                vstr = "%d..%d" % (v0, v)
                if dv > 1:
                    vstr += ":%d" % dv

        visitSummary.append(vstr)

    return "^".join(visitSummary)


def queryRegistry(field=None, visit=None, filterName=None, summary=False):
    """Query an input registry"""
    where = []; vals = []
    if field:
        where.append('field like ?')
        vals.append(field.replace("*", "%"))
    if filterName:
        where.append('filter like ?')
        vals.append(filterName.replace("*", "%"))
    if visit:
        where.append("visit = ?")
        vals.append(visit)
    where = "WHERE " + " AND ".join(where) if where else ""

    query = """
SELECT max(field), visit, max(expTime), max(dateObs), arm, ccd
FROM raw
%s
GROUP BY visit
ORDER BY max(expTime), visit
""" % (where)

    n = {}; expTimes = {}; visits = {}

    if registryFile.endswith('sqlite3'):
        conn = sqlite.connect(registryFile)
        isSqlite = True
    else:
        pgsqlConf = PgSqlConfig()
        pgsqlConf.load(registryFile)
        conn = pgsql.connect(host=pgsqlConf.host, port=pgsqlConf.port,
                             user=pgsqlConf.user, password=pgsqlConf.password,
                             database=pgsqlConf.db)
        isSqlite = False

    cursor = conn.cursor()

    if args.summary:
        print "%-20s %7s %s" % ("field", "expTime", "visit")
    else:
        print "%-20s %10s %7s %6s %3s %4s" % ("field", "dataObs", "expTime",
                                                   "visit", "arm", "ccd")

    if not isSqlite:
        query = query.replace("?", "%s")

    if isSqlite:
        ret = cursor.execute(query, vals)
    else:
        cursor.execute(query, vals)
        ret = cursor.fetchall()

    for line in ret:
        field, visit, expTime, dateObs, arm, ccd = line

        if summary:
            k = (field)
            if not n.get(k):
                n[k] = 0
                expTimes[k] = 0
                visits[k] = []

            n[k] += 1
            expTimes[k] += expTime
            visits[k].append(visit)
        else:
            print "%-20s %10s %7.1f %6d %3s %4d" % (field, dateObs, expTime,
                                                         visit, arm, ccd)

    conn.close()

    if summary:
        for k in sorted(n.keys()):
            field = k

            print "%-20s %7.1f %s" % (field, expTimes[k], formatVisits(visits[k]))

def queryCalibRegistry(what, filterName=None, summary=False):
    """Query a calib registry"""
    where = []; vals = []

    if filterName:
        where.append('filter like ?')
        vals.append(filterName.replace("*", "%"))
    where = "WHERE " + " AND ".join(where) if where else ""

    query = """
SELECT
    validStart, validEnd, calibDate, calibVersion, arm, ccd
FROM %s
%s
GROUP BY calibDate
ORDER BY calibDate
""" % (what, where)

    n = {}; expTimes = {}; visits = {}

    if registryFile.endswith('sqlite3'):
        conn = sqlite.connect(registryFile)
        isSqlite = True
    else:
        pgsqlConf = PgSqlConfig()
        pgsqlConf.load(registryFile)
        conn = pgsql.connect(host=pgsqlConf.host, port=pgsqlConf.port,
                           user=pgsqlConf.user, password=pgsqlConf.password,
                           database=pgsqlConf.db)
        isSqlite = False

    cursor = conn.cursor()

    if not isSqlite:
        query = query.replace("?", "%s")

    if summary:
        print >> sys.stderr, "No summary is available for calib data"
        return
    else:
        print "%-10s--%-10s  %-10s %-24s %3s %4s" % (
            "validStart", "validEnd", "calibDate", "calibVersion", "arm", "ccd")

    if isSqlite:
        ret = cursor.execute(query, vals)
    else:
        cursor.execute(query, vals)
        ret = cursor.fetchall()

    for line in ret:
        validStart, validEnd, calibDate, calibVersion, arm, ccd = line

        print "%-10s--%-10s  %-10s  %-24s %3s %4d" % (
            validStart, validEnd, calibDate, calibVersion, arm, ccd)

    conn.close()

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
Dump the contents of a registry

If no registry is provided, try $SUPRIME_DATA_DIR
""")

    parser.add_argument('registryFile', type=str, nargs="?", help="The registry or directory in question")
    parser.add_argument('--calib', type=str, help="The registration is a Calib; value is desired product")
    parser.add_argument('--field', type=str, help="Just tell me about this field (may be a glob with *)")
    parser.add_argument('--filter', dest="filterName", type=str,
                        help="Just tell me about this filter (may be a glob with *)")
    parser.add_argument('--verbose', action="store_true", help="How chatty should I be?", default=0)
    parser.add_argument('--visit', type=int, help="Just tell me about this visit")
    parser.add_argument('-s', '--summary', action="store_true", help="Print summary (grouped by field)")

    args = parser.parse_args()

    if not args.registryFile:
        args.registryFile = os.environ.get("SUPRIME_DATA_DIR", "")
        if args.calib:
            args.registryFile = os.path.join(args.registryFile, "CALIB")

    if os.path.exists(args.registryFile) and not os.path.isdir(args.registryFile):
        registryFile = args.registryFile
    else:
        registryFile = os.path.join(args.registryFile,
                                    "calibRegistry_pgsql.py" if args.calib else "registry_pgsql.py")
        if not os.path.exists(registryFile):
            registryFile = os.path.join(args.registryFile,
                                        "calibRegistry.sqlite3" if args.calib else "registry.sqlite3")
        if not os.path.exists(registryFile):
            print >> sys.stderr, "Unable to open %s" % registryFile
            sys.exit(1)

    if args.verbose:
        print "Reading %s" % registryFile

    if args.calib:
        queryCalibRegistry(args.calib, args.filterName, args.summary)
    else:
        queryRegistry(args.field, args.visit, args.filterName, args.summary)
