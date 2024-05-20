import random
import xlsxwriter
import time
import signal
import sys

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
            "Instances/lrc15-optimal-based.txt",
            "Instances/lrc15-location-based.txt",
            "Instances/lrc15-demand-based.txt",
            "Instances/lrc15-time-window-based.txt",
            "Instances/lrc15-location-and-servtime-and-time-window-based.txt",
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
            20.2
            ,
            31.8
            ,
            42.2
        ]
        self.EARLINESS_TARDINESS_PENALTY = [
            25
            ,
            45
            ,
            65
        ]

    def parse_route_math_model(self, solutionResult_MATH_MODEL, depotNode):
        resultList = list()
        for i in range(0, len(solutionResult_MATH_MODEL)):
            worksheetMathModelListItem = list()
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
            resultList.append(" => ".join(worksheetMathModelListItem))

        return " ; ".join(resultList)

    def parse_route_alns(self, solutionResult_ALNS):
        resultList = list()
        for k in range(0, len(solutionResult_ALNS)):
            if len(solutionResult_ALNS[k]) > 0:
                worksheetALNSListItem = list()
                vehicle_info_alns = f"Vehicle {k} - Trolley Count: {solutionResult_ALNS[k]['trolleyCount']}"
                route_alns = ' - '.join(solutionResult_ALNS[k]['route'])
                worksheetALNSListItem.append(vehicle_info_alns)
                worksheetALNSListItem.append(route_alns)
                resultList.append(" => ".join(worksheetALNSListItem))
        return " ; ".join(resultList)

    def parse_detail_excel(self, worksheetDetail, iterationNumber, trolleyCount, trolleyImpactRate, earlinessTardinessPenalty,
                           bestCost_MATH_MODEL,bestCost_ALNS, gap, cpuTime_MATH_MODEL, cpuTime_ALNS, routesDetailResult, solutionResult_ALNS, instance):
        worksheetDetail.append(
            ["ID", 'TROLLEY_COUNT', 'TROLLEY_IMPACT_RATE', 'EARLINESS/TARDINESS PENALTY'])
        worksheetDetail.append(["TEST_ID_" + str(iterationNumber), trolleyCount, trolleyImpactRate,
                                earlinessTardinessPenalty])
        worksheetDetail.append([instance, "Mathematical Formulation", "ALNS", "GAP"])
        worksheetDetail.append(["Result", round(bestCost_MATH_MODEL), round(bestCost_ALNS), gap])
        worksheetDetail.append(["cpuTime", cpuTime_MATH_MODEL, cpuTime_ALNS])
        worksheetDetail.append(["Mathematical Formulation"])
        for rd in range(0, len(routesDetailResult)):
            worksheetDetail.append(["", routesDetailResult[rd]])

        worksheetDetail.append(["ALNS Algorithm"])
        for k in range(0, len(solutionResult_ALNS)):
            if len(solutionResult_ALNS[k]) > 0:
                vehicle_info_alns = f"Vehicle {k} - Trolley Count: {solutionResult_ALNS[k]['trolleyCount']}"
                worksheetDetail.append(["", vehicle_info_alns])
                for rda in range(0, len(solutionResult_ALNS[k]["routeDetail"])):
                    worksheetDetail.append(["", solutionResult_ALNS[k]["routeDetail"][rda]])

        worksheetDetail.append([])

    def execute(self):
        start_time = time.time()
        print("Scenario Started")
        iterationNumber = 0
        iterationCount = 4000
        capacityOfTrolley = 60
        timeLimit = 60000 * 20



        for ins in range(0, len(self.INSTANCES)):
            infeasibleData = list()
            nonOptimalSolution = list()
            instance = self.INSTANCES[ins]
            file_name = instance.split('/')[-1].replace('.txt', '')
            workbook = Workbook()
            workbookDetail = Workbook()
            workbook.remove(workbook.active)  # Remove default sheet
            worksheet = workbook.create_sheet(title=f'Test Result')
            worksheet.append(
                ["", "TEST INSTANCES", '', '', '',"", "MATHEMATICAL FORMULATION", "", "", "", "ALNS", "", "", "GAP"])
            worksheet.append(
                ["TEST ID", "FILE NAME", "VEHICLE COUNT", 'TROLLEY COUNT', 'TROLLEY IMPACT RATE', 'EARLINESS/TARDINESS PENALTY',
                 "Math Form Is Optimal OR Feasible", "CPU TIME", "RESULT", "ROUTES", "CPU TIME", "RESULT", "ROUTES", ""])
            workbookDetail.remove(workbookDetail.active)  # Remove default sheet
            for vc in range(0, len(self.VEHICLE_COUNT)):
                vehicleCount = self.VEHICLE_COUNT[vc]
                vehicle_count = self.VEHICLE_COUNT[vc]
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

                            gap = ((round(bestCost_ALNS) - round(bestCost_MATH_MODEL)) / round(bestCost_ALNS)) * 100
                            if gap > 1 or gap < 0:
                                infeasibleData.append(
                                    ["TEST_ID: " + str(iterationNumber), "INSTANCE: " + str(
                                        instance) + ", VEHICLE_COUNT: " + str(
                                        vehicleCount) + ", TROLLEY_COUNT: " + str(
                                        trolleyCount) + ", TROLLEY_IMPACT_RATE: " + str(
                                        trolleyImpactRate) + ", EARLINESS_TARDINESS_PENALTY: " + str(
                                        earlinessTardinessPenalty), gap])

                            math_model_routes = self.parse_route_math_model(solutionResult_MATH_MODEL, depotNode)
                            heuristic_routes = self.parse_route_alns(solutionResult_ALNS)
                            worksheet.append(
                                [iterationNumber, file_name, vehicleCount, trolleyCount, trolleyImpactRate, earlinessTardinessPenalty, optimalOrFeasible,
                                 cpuTime_MATH_MODEL, round(bestCost_MATH_MODEL), math_model_routes, cpuTime_ALNS, round(bestCost_ALNS), heuristic_routes, gap])

                            self.parse_detail_excel(worksheetDetail, iterationNumber, trolleyCount,
                                                   trolleyImpactRate, earlinessTardinessPenalty,
                                                   bestCost_MATH_MODEL, bestCost_ALNS, gap, cpuTime_MATH_MODEL,
                                                   cpuTime_ALNS, routesDetailResult, solutionResult_ALNS, instance)

            workbook.save(f'excels/{file_name}.xlsx')
            workbookDetail.save(f'excels/details/{file_name}-details.xlsx')

            if len(infeasibleData) > 0:
                workbookInFeasible = Workbook()
                workbookInFeasible.remove(workbookInFeasible.active)
                worksheetInFeasible = workbookInFeasible.create_sheet(title="Gap Data")
                worksheetInFeasible.append(["TEST_ID", "TEST INSTANCE", "GAP"])
                for inf in range(0, len(infeasibleData)):
                    worksheetInFeasible.append(infeasibleData[inf])
                workbookInFeasible.save(f'excels/gap/{file_name}-gap.xlsx')

            if len(nonOptimalSolution) > 0:
                workbookNonOptimal = Workbook()
                workbookNonOptimal.remove(workbookNonOptimal.active)
                worksheetNonOptimal = workbookNonOptimal.create_sheet(title="Non-optimal Data")
                worksheetInFeasible.append(["TEST_ID", "TEST INSTANCE"])
                for f in range(0, len(nonOptimalSolution)):
                    worksheetNonOptimal.append(nonOptimalSolution[f])
                workbookNonOptimal.save(f'excels/non-optimal/{file_name}-non-optimal.xlsx')


        print("Scenario finished")
        endtime = time.time()
        cpuTime = round(endtime - start_time, 3)

        print("cpuTime: " + str(cpuTime) + " seconds")





scenario = ScenarioAnalysis()

scenario.execute()
