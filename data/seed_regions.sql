-- Static seed data: Brazil's 5 official geographic regions and their states.
-- No randomness/generation needed — these are fixed real-world facts.
-- Safe to run multiple times.

INSERT INTO regions (region_id, region_name) VALUES
    (1, 'North'),
    (2, 'Northeast'),
    (3, 'Central-West'),
    (4, 'Southeast'),
    (5, 'South')
ON CONFLICT (region_id) DO NOTHING;

INSERT INTO state_region_map (state_code, region_id) VALUES
    -- North
    ('AC', 1), ('AP', 1), ('AM', 1), ('PA', 1), ('RO', 1), ('RR', 1), ('TO', 1),
    -- Northeast
    ('AL', 2), ('BA', 2), ('CE', 2), ('MA', 2), ('PB', 2), ('PE', 2), ('PI', 2), ('RN', 2), ('SE', 2),
    -- Central-West
    ('DF', 3), ('GO', 3), ('MT', 3), ('MS', 3),
    -- Southeast
    ('ES', 4), ('MG', 4), ('RJ', 4), ('SP', 4),
    -- South
    ('PR', 5), ('RS', 5), ('SC', 5)
ON CONFLICT (state_code) DO NOTHING;