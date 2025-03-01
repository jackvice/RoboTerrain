// Copyright 2023 Fictionlab sp. z o.o.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifdef USE_IGNITION

#include <ignition/gazebo/System.hh>
#include <ignition/gazebo/Model.hh>
#include <ignition/gazebo/components.hh>
#include <ignition/plugin/Register.hh>

namespace gazebo = ignition::gazebo;

#define GZ_ADD_PLUGIN IGNITION_ADD_PLUGIN

#else

#define SUPPRESS_IGNITION_HEADER_DEPRECATION
#include <gz/sim/System.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/components.hh>
#include <gz/plugin/Register.hh>

namespace gazebo = gz::sim;

#endif

namespace leo_gz
{

class DifferentialSystem
  : public gazebo::System,
  public gazebo::ISystemConfigure,
  public gazebo::ISystemUpdate
{
  /// Model entity
  gazebo::Model model_{gazebo::kNullEntity};

  /// Force multiplier constant
  double force_constant_;

  /// Joint entities
  gazebo::Entity joint_a_, joint_b_;

  /// Whether the system has been properly configured
  bool configured_{false};

public:
  void Configure(
    const gazebo::Entity & entity,
    const std::shared_ptr<const sdf::Element> & sdf,
    gazebo::EntityComponentManager & ecm,
    gazebo::EventManager & /*eventMgr*/) override
  {
    model_ = gazebo::Model(entity);

    if (!model_.Valid(ecm)) {
      ignerr << "DifferentialSystem plugin should be attached to a model "
             << "entity. Failed to initialize." << std::endl;
      return;
    }

    if (!sdf->HasElement("jointA")) {
      ignerr << "No jointA element present. DifferentialSystem could not be loaded." << std::endl;
      return;
    }
    auto joint_a_name_ = sdf->Get<std::string>("jointA");

    if (!sdf->HasElement("jointB")) {
      ignerr << "No jointB element present. DifferentialSystem could not be loaded." << std::endl;
      return;
    }
    auto joint_b_name_ = sdf->Get<std::string>("jointB");

    if (!sdf->HasElement("forceConstant")) {
      ignerr << "No forceConstant element present. DifferentialSystem could not be loaded." <<
        std::endl;
      return;
    }
    force_constant_ = sdf->Get<double>("forceConstant");

    joint_a_ = model_.JointByName(ecm, joint_a_name_);
    if (joint_a_ == gazebo::kNullEntity) {
      ignerr << "Failed to find joint named \'" << joint_a_name_ << "\'" << std::endl;
      return;
    }

    joint_b_ = model_.JointByName(ecm, joint_b_name_);
    if (joint_b_ == gazebo::kNullEntity) {
      ignerr << "Failed to find joint named \'" << joint_b_name_ << "\'" << std::endl;
      return;
    }

    if (!ecm.EntityHasComponentType(joint_a_, gazebo::components::JointPosition().TypeId())) {
      ignmsg << "Joint A does not have JointPosition component. Creating one..." << std::endl;
      ecm.CreateComponent(joint_a_, gazebo::components::JointPosition());
    }

    if (!ecm.EntityHasComponentType(joint_b_, gazebo::components::JointPosition().TypeId())) {
      ignmsg << "Joint B does not have JointPosition component. Creating one..." << std::endl;
      ecm.CreateComponent(joint_b_, gazebo::components::JointPosition());
    }

    if (!ecm.EntityHasComponentType(joint_a_, gazebo::components::JointForceCmd().TypeId())) {
      ignmsg << "Joint A does not have JointForceCmd component. Creating one..." << std::endl;
      ecm.CreateComponent(joint_a_, gazebo::components::JointForceCmd({0}));
    }

    if (!ecm.EntityHasComponentType(joint_b_, gazebo::components::JointForceCmd().TypeId())) {
      ignmsg << "Joint B does not have JointForceCmd component. Creating one..." << std::endl;
      ecm.CreateComponent(joint_b_, gazebo::components::JointForceCmd({0}));
    }

    configured_ = true;
  }

  void Update(
    const gazebo::UpdateInfo & info,
    gazebo::EntityComponentManager & ecm) override
  {
    if (!configured_ || info.paused) {return;}

    // Retrieve components
    auto pos_a_component = ecm.Component<gazebo::components::JointPosition>(joint_a_);
    auto pos_b_component = ecm.Component<gazebo::components::JointPosition>(joint_b_);
    auto force_cmd_a_component = ecm.Component<gazebo::components::JointForceCmd>(joint_a_);
    auto force_cmd_b_component = ecm.Component<gazebo::components::JointForceCmd>(joint_b_);

    double pos_a = pos_a_component->Data()[0];
    double pos_b = pos_b_component->Data()[0];
    double angle_diff = pos_a - pos_b;

    double current_cmd_a = force_cmd_a_component->Data()[0];
    double current_cmd_b = force_cmd_b_component->Data()[0];

    *force_cmd_a_component = gazebo::components::JointForceCmd(
      {current_cmd_a - angle_diff * force_constant_});
    *force_cmd_b_component = gazebo::components::JointForceCmd(
      {current_cmd_b + angle_diff * force_constant_});
  }
};

}  // namespace leo_gz

GZ_ADD_PLUGIN(
  leo_gz::DifferentialSystem,
  gazebo::System,
  leo_gz::DifferentialSystem::ISystemConfigure,
  leo_gz::DifferentialSystem::ISystemUpdate)
