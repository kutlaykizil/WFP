/* REGULAR CHECKS */

/* List mismatches between wf and wf_turbines based on their id
excluding farms where MWe(isletmedeki kapasite) =! 0 */
with tmp as(
    select wf.id, wf."Tesis Adı",wf."İşletmedeki Kapasite (MWe)",wf_turbines.id, wf_turbines."proje adi"
from wf
         full outer join wf_turbines on wf.id = wf_turbines.id
where wf.id is null
   or wf_turbines.id is null
)
select * from tmp
where tmp."İşletmedeki Kapasite (MWe)" = 0 is false;

/* List wf (İşletmedeki Kapasite) <> wf_turbines incorrect powers (for MWe) */
SELECT t1.id,
       wf_MWm,
       wf_turbines_MWm,
       wf_MWe_for_referance
FROM (SELECT id,
             SUM("İşletmedeki Kapasite (MWm)") AS wf_MWm,
             SUM("İşletmedeki Kapasite (MWe)") AS wf_MWe_for_referance
      FROM wf
      GROUP BY id) AS t1
         LEFT JOIN (SELECT id,
                           SUM("installed power") AS wf_turbines_MWm
                    FROM wf_turbines
                    GROUP BY id) AS t2 ON t1.id = t2.id
WHERE t1.wf_MWm <> t2.wf_turbines_MWm;

/* Check if there any totally wrong projections in wt */
select *
from wf_turbine_coordinates
where um != 27
  AND um != 33
  AND um != 39
  AND um != 45;

/* Check if there any totally wrong projections in wf_border_coordinates */
select *
from wf_border_coordinates
where km != 27
  AND km != 33
  AND km != 39
  AND km != 45;

/* Check if there are any mismatches between the projections of wt and wf_border_coordinates geometries */
SELECT wf_turbine_coordinates.id, um, km
FROM wf_turbine_coordinates
         INNER JOIN (SELECT id, km FROM wf_border_coordinates GROUP BY id, km) as B ON B.id = wf_turbine_coordinates.id
WHERE NOT (um = km)
group by wf_turbine_coordinates.id, um, km
order by id;

/* Check if there is a unique id for epias id data in wf and list non-uniques */
SELECT id,"Tesis Adı","Lisans No","Lisans Sahibi",epias_id,epias_name FROM wf
WHERE epias_id IN (
    SELECT epias_id
    FROM wf
    GROUP BY epias_id
    HAVING COUNT(*) > 1
)
and epias_id IS NOT NULL
order by epias_name;

/* Check if there is a change in # of empty epias_id rows from last import run */
select "Tesis Adı" from wf
where epias_id is null and id is not null;

/* Check if a plant with revoked license still has an epias_id defined
   and there is another plant with a similar name without epian_id*/
with tmp as (
    select "Tesis Adı",id,epias_name,id, "Lisans Durumu"
    from wf
    where "Lisans Durumu" = 'Sonlandırıldı'
    and epias_id is not null
)
select tmp.epias_name,wf."Tesis Adı",wf.id from tmp
left join wf on tmp."Tesis Adı" ilike wf."Tesis Adı"
where wf."Lisans Durumu" = 'Yürürlükte';

/* See the plants left out in wf */
select * from wf
where wf.epias_id is NULL and "Lisans Durumu" = 'Yürürlükte';

/* Check the sum of old prod and new prod imported with https-json for every id (takes about 5 minutes) */
WITH id_sequence AS (
  SELECT generate_series(1, 307) AS id
)
SELECT id,
       (SELECT SUM("prod(MWh)") FROM prod WHERE id = id_sequence.id) -
       (SELECT SUM("production (MWh)") FROM "productionss-2022-12-31" WHERE id = id_sequence.id)
       AS difference
FROM id_sequence;







/* PROD TABLE RECREATION */

/* Recreate prod table because we don't have a way of adding new data from the last date
This would not be an issue with the EPIAS API if we got access*/
drop table if exists prod;
create table prod
(
    id          int,
    date        text,
    time        text,
    "prod(MWh)" text
);
/* RUN csv2prod.sql HERE */
/*
\include /Scripts/csv2prod.sql
*/
/* Change the delimiter from comma to dot because data source is Turkish */
update prod
set "prod(MWh)" = replace("prod(MWh)", ',', '.');
alter table prod
    alter column "prod(MWh)" type float using "prod(MWh)"::double precision;




/* Check the matches between ministry and epdk, AND tureb and epdk */
WITH moe_data AS (
    SELECT
        wf.wf_id,
        cast(sum(moe."İLAVE KURULU GÜÇ (MWe)") as numeric)as sum_moe,
        wf."İşletmedeki Kapasite (MWe)" as wf,
        CASE
            WHEN cast(sum(moe."İLAVE KURULU GÜÇ (MWe)") as numeric) = cast(wf."İşletmedeki Kapasite (MWe)" as numeric) THEN 'Match'
            ELSE 'No Match'
        END as flag
    FROM wf
    INNER JOIN ministry_of_energy moe ON wf.wf_id = moe.wf_id where wf.version = (select max(version) from wf)
    GROUP BY wf.wf_id, wf."İşletmedeki Kapasite (MWe)"
    ORDER BY wf.wf_id
),

wft_data AS (
    SELECT
      wf.wf_id,
      cast(sum(wft."installed power") as numeric) as sum_wft,
      cast(wf."Kurulu Güç (MWm)" as numeric) as wf_mwm,  -- Alias for clarity
      cast(wf."Kurulu Güç (MWe)" as numeric) as wf_mwe,  -- Include the new column
      CASE
        WHEN cast(sum(wft."installed power") as numeric) = cast(wf."Kurulu Güç (MWm)" as numeric)
          or cast(sum(wft."installed power") as numeric) = cast(wf."Kurulu Güç (MWe)" as numeric)  -- Extend the match
        THEN 'Match'
        ELSE 'No Match'
      END as flag
    FROM wf
    INNER JOIN wf_turbines wft ON wf.wf_id = wft.wf_id
    WHERE wf.version = (SELECT MAX(version) FROM wf)
    GROUP BY wf.wf_id, wf."Kurulu Güç (MWm)", wf."Kurulu Güç (MWe)"  -- Add new column to GROUP BY
    ORDER BY wf.wf_id
)

SELECT *
FROM wft_data
JOIN moe_data ON wft_data.wf_id = moe_data.wf_id
ORDER BY wft_data.wf_id;





/* GEOMETRY */

/* Only run the followings when there is a new wind farm entry */
/* Create point geometry for wt from lat-long. */
update wf_turbine_coordinates
set geom = st_transform((st_setsrid(st_makepoint(ux, uy),
                                    (CASE
                                         WHEN um = 27 THEN 23035
                                         WHEN um = 33 THEN 23036
                                         WHEN um = 39 THEN 23037
                                         WHEN um = 45 THEN 23038 END))), 3035);
/* Create point geometry for wf_border_coordinates from lat-long. */
update wf_border_coordinates
set geom_point = st_transform(st_setsrid(st_makepoint(kx, ky), (CASE
                                                                    WHEN km = 27 THEN 23035
                                                                    WHEN km = 33 THEN 23036
                                                                    WHEN km = 39 THEN 23037
                                                                    WHEN km = 45 THEN 23038 END)), 3035);
/* Create linestring geometry in a tmp table from point geometry */
select st_makeline(geom_point order by point_index)
    as geom_linestring, id
    into tmp
    from wf_border_coordinates
    group by id order by id;
/* Check if linestring is closed and add the start point to the end */
update tmp
set geom_linestring = ST_AddPoint(geom_linestring, ST_StartPoint(geom_linestring))
WHERE ST_IsClosed(geom_linestring) = false;
/* create polygon geometries for wf */
update wf
set geom_polygon = st_makepolygon(geom_linestring)
from tmp
where tmp.id=wf.id and st_numpoints(tmp.geom_linestring) > 3;
/* Delete tmp because we don't need it */
drop table tmp;
