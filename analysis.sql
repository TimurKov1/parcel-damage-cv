-- Test claims by damage type, coverage and routing outcome.
SELECT defect, route, COUNT(*) AS claims,
       SUM(CASE WHEN damaged = 1 AND route = 'no_visible_damage' THEN 1 ELSE 0 END)
           AS missed_damage,
       SUM(CASE WHEN damaged = 0 AND route = 'visible_damage' THEN 1 ELSE 0 END)
           AS false_damage
FROM test_claims
GROUP BY defect, route
ORDER BY defect, route;
