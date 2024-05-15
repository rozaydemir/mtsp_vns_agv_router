import random
import xlsxwriter
import time

import numpy as np

# from pdptw import PDPTW
from heuristic import Heuristic
from algorithm import Algorithm

import pandas as pd
import os
import time
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows


class ScenarioAnalysis:

    def __init__(self):
        self.INSTANCES = [
            # "Instances/lrc9-optimal-based.txt",
            # "Instances/lrc9-servtime-based.txt",
            # "Instances/lrc9-location-based.txt",
            "Instances/lrc9-time-window-based.txt",
            # "Instances/lrc9-demand-based.txt"
        ]
        self.VEHICLE_COUNT = [
            1
            ,
            2
            ,
            3
        ]
        self.TROLLEY_COUNT = [
            1
            ,
            2
            ,
            3
        ]
        self.TROLLEY_IMPACT_RATE = [
            1.2
            ,
            1.8
            ,
            2.2
        ]
        self.EARLINESS_TARDINESS_PENALTY = [
            25
            ,
            45
            ,
            65
        ]

    def execute(self):
        start_time = time.time()
        print("Scenario Started")
        iterationNumber = 0
        iterationCount = 3000
        infeasibleData = list()
        nonOptimalSolution = list()
        capacityOfTrolley = 60
        timeLimit = 60000 * 5

        for ins in range(0, len(self.INSTANCES)):
            instance = self.INSTANCES[ins]
            file_name = instance.split('/')[-1].replace('.txt', '')
            workbook = Workbook()
            workbookDetail = Workbook()
            workbook.remove(workbook.active)  # Remove default sheet
            workbookDetail.remove(workbookDetail.active)  # Remove default sheet
            for vc in range(0, len(self.VEHICLE_COUNT)):
                vehicleCount = self.VEHICLE_COUNT[vc]
                vehicle_count = self.VEHICLE_COUNT[vc]
                worksheet = workbook.create_sheet(title=f'{vehicle_count} Vehicle')
                worksheetDetail = workbookDetail.create_sheet(title=f'{vehicle_count} Vehicle')
                for tc in range(0, len(self.TROLLEY_COUNT)):
                    trolleyCount = self.TROLLEY_COUNT[tc]
                    for tir in range(0, len(self.TROLLEY_IMPACT_RATE)):
                        trolleyImpactRate = self.TROLLEY_IMPACT_RATE[tir]
                        for etp in range(0, len(self.EARLINESS_TARDINESS_PENALTY)):
                            iterationNumber += 1
                            earlinessTardinessPenalty = self.EARLINESS_TARDINESS_PENALTY[etp]
                            print("===============================================================================")
                            print("Iteration Number : " + str(iterationNumber))
                            print(
                                f"INSTANCE: {instance}, VEHICLE_COUNT : {vehicleCount}, TROLLEY_COUNT: {trolleyCount}"
                                f", TROLLEY_IMPACT_RATE: {trolleyImpactRate}, EARLINESS_TARDINESS_PENALTY: {earlinessTardinessPenalty}")

                            algorithm = Algorithm(instance, vehicleCount, capacityOfTrolley, trolleyCount,
                                                  trolleyImpactRate, earlinessTardinessPenalty,
                                                  earlinessTardinessPenalty, timeLimit)
                            optimalOrFeasible, bestCost_MATH_MODEL, cpuTime_MATH_MODEL, solutionResult_MATH_MODEL, depotNode, routesDetailResult = algorithm.execute()

                            if optimalOrFeasible == "NON-OPTIMAL":
                                nonOptimalSolution.append(
                                    ["TEST_ID: " + str(iterationNumber),
                                     "INSTANCE: " + str(
                                         instance) + ", VEHICLE_COUNT: " + str(
                                         vehicleCount) + ", TROLLEY_COUNT: " + str(
                                         trolleyCount) + ", TROLLEY_IMPACT_RATE: " + str(
                                         trolleyImpactRate) + ", EARLINESS_TARDINESS_PENALTY: " + str(
                                         earlinessTardinessPenalty)
                                     ])
                                continue

                            heuristic = Heuristic(instance, vehicleCount, capacityOfTrolley, trolleyCount,
                                                  trolleyImpactRate, earlinessTardinessPenalty,
                                                  earlinessTardinessPenalty, iterationCount, bestCost_MATH_MODEL)
                            bestCost_ALNS, cpuTime_ALNS, solutionResult_ALNS = heuristic.execute()

                            worksheet.append(
                                ["TEST_ID", 'TROLLEY_COUNT', 'TROLLEY_IMPACT_RATE', 'EARLINESS/TARDINESS PENALTY'])
                            worksheetDetail.append(
                                ["ID", 'TROLLEY_COUNT', 'TROLLEY_IMPACT_RATE', 'EARLINESS/TARDINESS PENALTY'])
                            worksheet.append(
                                [iterationNumber, trolleyCount, trolleyImpactRate, earlinessTardinessPenalty])
                            worksheetDetail.append(["TEST_ID_" + str(iterationNumber), trolleyCount, trolleyImpactRate,
                                                    earlinessTardinessPenalty])
                            worksheet.append([instance, "Mathematical Formulation", "ALNS", "GAP"])
                            worksheetDetail.append([instance, "Mathematical Formulation", "ALNS", "GAP"])

                            gap = ((round(bestCost_ALNS) - round(bestCost_MATH_MODEL)) / round(bestCost_ALNS)) * 100
                            if gap > 1 or gap < 0:
                                infeasibleData.append(
                                    ["TEST_ID: " + str(iterationNumber), "INSTANCE: " + str(
                                        instance) + ", VEHICLE_COUNT: " + str(
                                        vehicleCount) + ", TROLLEY_COUNT: " + str(
                                        trolleyCount) + ", TROLLEY_IMPACT_RATE: " + str(
                                        trolleyImpactRate) + ", EARLINESS_TARDINESS_PENALTY: " + str(
                                        earlinessTardinessPenalty), gap])

                            worksheet.append(["Result", round(bestCost_MATH_MODEL), round(bestCost_ALNS), gap])
                            worksheetDetail.append(["Result", round(bestCost_MATH_MODEL), round(bestCost_ALNS), gap])
                            worksheet.append(["cpuTime", cpuTime_MATH_MODEL, cpuTime_ALNS])
                            worksheetDetail.append(["cpuTime", cpuTime_MATH_MODEL, cpuTime_ALNS])
                            worksheet.append(["Math Form Is Optimal OR Feasible", optimalOrFeasible])
                            worksheet.append(["Vehicles / Routes"])

                            worksheetDetail.append(["Mathematical Formulation"])
                            for rd in range(0, len(routesDetailResult)):
                                worksheetDetail.append(["", routesDetailResult[rd]])

                            worksheetMathModelListItem = list()
                            worksheetALNSListItem = list()
                            for i in range(0, len(solutionResult_MATH_MODEL)):
                                vehicle_info = f"Vehicle {i} - Trolley Count: {solutionResult_MATH_MODEL[i]['trolleyCount']}"
                                route_math_model = ' - '.join(solutionResult_MATH_MODEL[i]['route'])

                                pairs = route_math_model.split(" - ")
                                pairs = [eval(pair) for pair in pairs]
                                route_map = {}

                                for start, end in pairs:
                                    route_map[start] = end

                                start_point = pairs[0][0]
                                for start, end in pairs:
                                    if start not in route_map.values():
                                        start_point = start
                                        break

                                route = ["D0"]
                                current = start_point
                                while current in route_map:
                                    next_point = route_map[current]
                                    if next_point == depotNode:
                                        route.append("D0")
                                    else:
                                        route.append(f"C{next_point}")
                                    current = next_point

                                route_str = " - ".join(map(str, route))
                                worksheetMathModelListItem.append(vehicle_info)
                                worksheetMathModelListItem.append(route_str)

                            worksheetDetail.append(["ALNS Algorithm"])
                            for k in range(0, len(solutionResult_ALNS)):
                                if len(solutionResult_ALNS[k]) > 0:
                                    vehicle_info_alns = f"Vehicle {k} - Trolley Count: {solutionResult_ALNS[k]['trolleyCount']}"
                                    route_alns = ' - '.join(solutionResult_ALNS[k]['route'])
                                    # worksheet.append([vehicle_info_alns, route_alns])
                                    worksheetALNSListItem.append(vehicle_info_alns)
                                    worksheetALNSListItem.append(route_alns)
                                    worksheetDetail.append(["", vehicle_info_alns])
                                    for rda in range(0, len(solutionResult_ALNS[k]["routeDetail"])):
                                        worksheetDetail.append(["", solutionResult_ALNS[k]["routeDetail"][rda]])

                            max_length = max(len(worksheetMathModelListItem), len(worksheetALNSListItem))
                            for m in range(0, max_length):
                                alnsItem = ""
                                mathModelItem = ""
                                if 0 <= m < len(worksheetMathModelListItem):
                                    mathModelItem = worksheetMathModelListItem[m]

                                if 0 <= m < len(worksheetALNSListItem):
                                    alnsItem = worksheetALNSListItem[m]
                                worksheet.append(["", mathModelItem, alnsItem])

                            worksheet.append([])
                            worksheetDetail.append([])

            workbook.save(f'excels/{file_name}.xlsx')
            workbookDetail.save(f'excels/details/{file_name}-details.xlsx')

        if len(infeasibleData) > 0:
            workbookInFeasible = Workbook()
            workbookInFeasible.remove(workbookInFeasible.active)
            worksheetInFeasible = workbookInFeasible.create_sheet(title="Gap Data")
            for id in range(0, len(infeasibleData)):
                worksheetInFeasible.append(nonOptimalSolution[id])
            workbookInFeasible.save(f'excels/gap/{file_name}-gap.xlsx')

        if len(nonOptimalSolution) > 0:
            workbookNonOptimal = Workbook()
            workbookNonOptimal.remove(workbookNonOptimal.active)
            worksheetNonOptimal = workbookNonOptimal.create_sheet(title="Non-optimal Data")
            for f in range(0, len(nonOptimalSolution)):
                worksheetNonOptimal.append([nonOptimalSolution[f]])
            workbookNonOptimal.save(f'excels/non-optimal/{file_name}-non-optimal.xlsx')

        print("Scenario finished")
        endtime = time.time()
        cpuTime = round(endtime - start_time, 3)

        print("cpuTime: " + str(cpuTime) + " seconds")


scenario = ScenarioAnalysis()

scenario.execute()
