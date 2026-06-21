# config.py
# Shared configuration for the AI Traffic Sentinel pipeline

VIOLATION_STYLES = {
    'triple_riding': {'color': (0, 255, 255), 'label': 'Triple Riding'},
    'helmet_non_compliance': {'color': (255, 0, 0), 'label': 'No Helmet'},
    'illegal_parking': {'color': (0, 0, 255), 'label': 'Illegal Parking'},
    'license_plate': {'color': (0, 255, 0), 'label': 'License Plate'},
}

DEFAULT_CONF = {
    'triple_riding': 0.45,
    'helmet_non_compliance': 0.40,
    'illegal_parking': 0.40,
    'license_plate': 0.40,
}

MODEL_RUN_CONF = {
    'triple_riding': 0.25,
    'helmet_non_compliance': 0.20,
    'illegal_parking': 0.40,
    'license_plate': 0.25,
}

TWO_WHEELER_CLASSES = {'motorcycle', 'bicycle'}
NON_RIDER_VEHICLE_CLASSES = {'car', 'bus', 'truck'}

DEDUP_IOU = 0.45
RIDER_GATE_OVERLAP = 0.08
PLATE_VEHICLE_OVERLAP = 0.10