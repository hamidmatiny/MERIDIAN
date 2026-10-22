import pandas as pd

def quarantine(df):
    """
    Quarantine rows where total_distance_km is null.
    
    The incident shows a NULL_EXPLOSION in total_distance_km caused by a backfill
    that wrote null values. Since nulls cannot be reconstructed and the column has
    a not_nullable constraint, we must drop these rows.
    """
    return df['total_distance_km'].isna()
