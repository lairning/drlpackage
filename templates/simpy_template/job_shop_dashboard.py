

import streamlit as st
import altair as alt
import pandas as pd
import json
from lairningdecisions.utils.db import db_connect, TRAINER_DB_NAME, P_MARKER

DBTYPE = 'sqlite'  # if platform.system() == 'Windows' else 'mysql'

P_MARKER = "?" if DBTYPE == 'sqlite' else "%s"

SQLParamList = lambda n: '' if n <= 0 else (P_MARKER + ',') * (n - 1) + P_MARKER

TRAINER_DB = db_connect(TRAINER_DB_NAME)

def get_schedule_events(policy_id: int, simulation_id: int, left:int, right:int):
    sql = '''select info from policy_run_info 
             where event_type = "schedule" and policy_id = {} and simulation_id = {} 
                   and action_step >= {} and action_step <= {}'''.format(
        P_MARKER, P_MARKER, P_MARKER, P_MARKER)
    df = pd.read_sql_query(sql, TRAINER_DB, params=(policy_id, simulation_id, left, right))
    return pd.DataFrame([ json.loads(x) for x in df['info']])

def get_demand(policy_id: int, simulation_id: int, left:int, right:int):
    sql = '''select action_step, info from policy_run_info 
             where event_type = "demand" and policy_id = {} and simulation_id = {} 
                   and action_step >= {} and action_step <= {}'''.format(
        P_MARKER, P_MARKER, P_MARKER, P_MARKER)
    df = pd.read_sql_query(sql, TRAINER_DB, params=(policy_id, simulation_id, left, right))
    df = pd.DataFrame([ {**json.loads(df.loc[i,:]['info']), **{'time':df.loc[i,:]['action_step']}} for i in df.index ])
    return df

def get_stock_cost(policy_id: int, simulation_id: int, left:int, right:int):
    sql = '''select action_step, info from policy_run_info 
             where event_type = "stock_cost" and policy_id = {} and simulation_id = {} 
                   and action_step >= {} and action_step <= {}'''.format(
        P_MARKER, P_MARKER, P_MARKER, P_MARKER)
    df = pd.read_sql_query(sql, TRAINER_DB, params=(policy_id, simulation_id, left, right))
    df = pd.DataFrame([ {**json.loads(df.loc[i,:]['info']), **{'time':df.loc[i,:]['action_step']}} for i in df.index ])
    return df

policy_id = st.sidebar.selectbox('Which Policy?',(1,2))

st.title('Machine Scheduling')

left, right = st.slider('START TIME', value=(0,600))

schedule_data = get_schedule_events(policy_id=policy_id, simulation_id=0, left=left, right=right)

demand_data = get_demand(policy_id=policy_id, simulation_id=0, left=left, right=right)

stock_cost = get_stock_cost(policy_id=policy_id, simulation_id=0, left=left, right=right)

chart_width = 800
    
gant_chart = alt.Chart(schedule_data).mark_bar().encode(
    x = alt.X('start:Q', scale=alt.Scale(zero=False)),
    x2='stop:Q',
    y='start_time:O',
    color='product:N'
).properties(
    width=chart_width,
    height=400
)

revenue_chart = alt.Chart(demand_data[demand_data['satisfied']==True]).mark_bar().encode(
    x=alt.X('time:Q', scale=alt.Scale(zero=False)),
    y=alt.Y('sum(revenue)', axis=alt.Axis(title='Revenue')),
    color='product:N'
).properties(
    width=chart_width,
    height=200
)

lost_revenue_chart = alt.Chart(demand_data[demand_data['satisfied']==False]).mark_bar().encode(
    x=alt.X('time:Q', scale=alt.Scale(zero=False)),
    y=alt.Y('sum(revenue)', axis=alt.Axis(title='Lost Revenue')),
    color='product:N'
).properties(
    width=chart_width,
    height=200
)

stock_cost_chart = alt.Chart(stock_cost).mark_bar().encode(
    x=alt.X('time:Q', scale=alt.Scale(zero=False)),
    y=alt.Y('sum(cost)', axis=alt.Axis(title='Stock Cost')),
    color='product:N'
).properties(
    width=chart_width,
    height=200
)

st.write(gant_chart)
st.write(revenue_chart)
st.write(lost_revenue_chart)
st.write(stock_cost_chart)


